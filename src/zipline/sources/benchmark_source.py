#
# Copyright 2018 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd

from zipline.errors import (
    InvalidBenchmarkAsset,
    BenchmarkAssetNotAvailableTooEarly,
    BenchmarkAssetNotAvailableTooLate,
)


class BenchmarkSource:
    def __init__(
        self,
        benchmark_asset,
        trading_calendar,
        sessions,
        data_portal,
        emission_rate="daily",
        benchmark_returns=None,
    ):
        self.benchmark_asset = benchmark_asset
        self.sessions = sessions
        self.emission_rate = emission_rate
        self.data_portal = data_portal

        if len(sessions) == 0:
            self._precalculated_series = pd.Series()
        elif benchmark_asset is not None:
            self._validate_benchmark(benchmark_asset)
            (
                self._precalculated_series,
                self._daily_returns,
            ) = self._initialize_precalculated_series(
                benchmark_asset, trading_calendar, sessions, data_portal
            )
        elif benchmark_returns is not None:
            self._daily_returns = daily_series = benchmark_returns.reindex(
                sessions,
            ).fillna(0)

            if self.emission_rate == "minute":
                # we need to take the env's benchmark returns, which are daily,
                # and resample them to minute
                minutes = trading_calendar.sessions_minutes(sessions[0], sessions[-1])
                minute_series = daily_series.tz_localize(minutes.tzinfo).reindex(
                    index=minutes, method="ffill"
                )

                self._precalculated_series = minute_series
            else:
                self._precalculated_series = daily_series
        else:
            raise Exception(
                "Must provide either benchmark_asset or " "benchmark_returns."
            )

    def get_value(self, dt):
        """Look up the returns for a given dt.

        Parameters
        ----------
        dt : datetime
            The label to look up.

        Returns
        -------
        returns : float
            The returns at the given dt or session.

        See Also
        --------
        :class:`zipline.sources.benchmark_source.BenchmarkSource.daily_returns`

        .. warning::

           This method expects minute inputs if ``emission_rate == 'minute'``
           and session labels when ``emission_rate == 'daily``.
        """
        return self._precalculated_series.loc[dt]

    def get_range(self, start_dt, end_dt):
        """Look up the returns for a given period.

        Parameters
        ----------
        start_dt : datetime
            The inclusive start label.
        end_dt : datetime
            The inclusive end label.

        Returns
        -------
        returns : pd.Series
            The series of returns.

        See Also
        --------
        :class:`zipline.sources.benchmark_source.BenchmarkSource.daily_returns`

        .. warning::

           This method expects minute inputs if ``emission_rate == 'minute'``
           and session labels when ``emission_rate == 'daily``.
        """
        return self._precalculated_series.loc[start_dt:end_dt]

    def daily_returns(self, start, end=None):
        """Returns the daily returns for the given period.

        Parameters
        ----------
        start : datetime
            The inclusive starting session label.
        end : datetime, optional
            The inclusive ending session label. If not provided, treat
            ``start`` as a scalar key.

        Returns
        -------
        returns : pd.Series or float
            The returns in the given period. The index will be the trading
            calendar in the range [start, end]. If just ``start`` is provided,
            return the scalar value on that day.
        """
        if end is None:
            return self._daily_returns[start]

        return self._daily_returns[start:end]

    def _validate_benchmark(self, benchmark_asset):
        # check if this security has a stock dividend.  if so, raise an
        # error suggesting that the user pick a different asset to use
        # as benchmark.
        stock_dividends = self.data_portal.get_stock_dividends(
            self.benchmark_asset, self.sessions
        )

        if len(stock_dividends) > 0:
            raise InvalidBenchmarkAsset(
                sid=str(self.benchmark_asset), dt=stock_dividends[0]["ex_date"]
            )

        if benchmark_asset.start_date > self.sessions[0]:
            # the asset started trading after the first simulation day
            raise BenchmarkAssetNotAvailableTooEarly(
                sid=str(self.benchmark_asset),
                dt=self.sessions[0],
                start_dt=benchmark_asset.start_date,
            )

        if benchmark_asset.end_date < self.sessions[-1]:
            # the asset stopped trading before the last simulation day
            raise BenchmarkAssetNotAvailableTooLate(
                sid=str(self.benchmark_asset),
                dt=self.sessions[-1],
                end_dt=benchmark_asset.end_date,
            )

    @staticmethod
    def _compute_daily_returns(g):
        return (g[-1] - g[0]) / g[0]

    @classmethod
    def downsample_minute_return_series(cls, trading_calendar, minutely_returns):
        sessions = trading_calendar.minutes_to_sessions(
            minutely_returns.index,
        )
        closes = trading_calendar.closes[sessions[0] : sessions[-1]]
        daily_returns = minutely_returns[closes].pct_change()
        daily_returns.index = closes.index
        return daily_returns.iloc[1:]

    def _initialize_precalculated_series(
        self, asset, trading_calendar, trading_days, data_portal
    ):
        """
        Internal method that pre-calculates the benchmark return series for
        use in the simulation.

        Parameters
        ----------
        asset:  Asset to use

        trading_calendar: TradingCalendar

        trading_days: pd.DateTimeIndex

        data_portal: DataPortal

        Notes
        -----
        If the benchmark asset started trading after the simulation start,
        or finished trading before the simulation end, exceptions are raised.

        If the benchmark asset started trading the same day as the simulation
        start, the first available minute price on that day is used instead
        of the previous close.

        We use history to get an adjusted price history for each day's close,
        as of the look-back date (the last day of the simulation).  Prices are
        fully adjusted for dividends, splits, and mergers.

        Returns
        -------
        returns : pd.Series
            indexed by trading day, whose values represent the %
            change from close to close.
        daily_returns : pd.Series
            the partial daily returns for each minute
        """
        if self.emission_rate == "minute":
            minutes = trading_calendar.sessions_minutes(
                self.sessions[0], self.sessions[-1]
            )
            benchmark_series = data_portal.get_history_window(
                [asset],
                minutes[-1],
                bar_count=len(minutes) + 1,
                frequency="1m",
                field="price",
                data_frequency=self.emission_rate,
                ffill=True,
            )[asset]

            return (
                benchmark_series.pct_change()[1:],
                self.downsample_minute_return_series(
                    trading_calendar,
                    benchmark_series,
                ),
            )

        start_date = asset.start_date
        if start_date < trading_days[0]:
            # get the window of close prices for benchmark_asset from the
            # last trading day of the simulation, going up to one day
            # before the simulation start day (so that we can get the %
            # change on day 1)
            benchmark_series = data_portal.get_history_window(
                [asset],
                trading_days[-1],
                bar_count=len(trading_days) + 1,
                frequency="1d",
                field="price",
                data_frequency=self.emission_rate,
                ffill=True,
            )[asset]

            returns = benchmark_series.pct_change()[1:]
            return returns, returns
        elif start_date == trading_days[0]:
            # Attempt to handle case where stock data starts on first
            # day, in this case use the open to close return.
            benchmark_series = data_portal.get_history_window(
                [asset],
                trading_days[-1],
                bar_count=len(trading_days),
                frequency="1d",
                field="price",
                data_frequency=self.emission_rate,
                ffill=True,
            )[asset]

            # get a minute history window of the first day
            first_open = data_portal.get_spot_value(
                asset,
                "open",
                trading_days[0],
                "daily",
            )
            first_close = data_portal.get_spot_value(
                asset,
                "close",
                trading_days[0],
                "daily",
            )

            first_day_return = (first_close - first_open) / first_open

            returns = benchmark_series.pct_change()[:]
            returns[0] = first_day_return
            return returns, returns
        else:
            raise ValueError(
                "cannot set benchmark to asset that does not exist during"
                " the simulation period (asset start date=%r)" % start_date
            )
