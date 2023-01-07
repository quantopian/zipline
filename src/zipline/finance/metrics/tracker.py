#
# Copyright 2017 Quantopian, Inc.
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
import logging

from ..ledger import Ledger
from zipline.utils.exploding_object import NamedExplodingObject


log = logging.getLogger(__name__)


class MetricsTracker:
    """The algorithm's interface to the registered risk and performance
    metrics.

    Parameters
    ----------
    trading_calendar : TrandingCalendar
        The trading calendar used in the simulation.
    first_session : pd.Timestamp
        The label of the first trading session in the simulation.
    last_session : pd.Timestamp
        The label of the last trading session in the simulation.
    capital_base : float
        The starting capital for the simulation.
    emission_rate : {'daily', 'minute'}
        How frequently should a performance packet be generated?
    data_frequency : {'daily', 'minute'}
        The data frequency of the data portal.
    asset_finder : AssetFinder
        The asset finder used in the simulation.
    metrics : list[Metric]
        The metrics to track.
    """

    _hooks = (
        "start_of_simulation",
        "end_of_simulation",
        "start_of_session",
        "end_of_session",
        "end_of_bar",
    )

    @staticmethod
    def _execution_open_and_close(calendar, session):
        if session.tzinfo is not None:
            session = session.tz_localize(None)

        open_ = calendar.session_first_minute(session)
        close = calendar.session_close(session)

        execution_open = open_
        execution_close = close

        return execution_open, execution_close

    def __init__(
        self,
        trading_calendar,
        first_session,
        last_session,
        capital_base,
        emission_rate,
        data_frequency,
        asset_finder,
        metrics,
    ):
        self.emission_rate = emission_rate

        self._trading_calendar = trading_calendar
        self._first_session = first_session
        self._last_session = last_session
        self._capital_base = capital_base
        self._asset_finder = asset_finder

        self._current_session = first_session
        self._market_open, self._market_close = self._execution_open_and_close(
            trading_calendar,
            first_session,
        )
        self._session_count = 0

        self._sessions = sessions = trading_calendar.sessions_in_range(
            first_session,
            last_session,
        )
        self._total_session_count = len(sessions)

        self._ledger = Ledger(sessions, capital_base, data_frequency)

        self._benchmark_source = NamedExplodingObject(
            "self._benchmark_source",
            "_benchmark_source is not set until ``handle_start_of_simulation``"
            " is called",
        )

        if emission_rate == "minute":

            def progress(self):
                return 1.0  # a fake value

        else:

            def progress(self):
                return self._session_count / self._total_session_count

        # don't compare these strings over and over again!
        self._progress = progress

        # bind all of the hooks from the passed metric objects.
        for hook in self._hooks:
            registered = []
            for metric in metrics:
                try:
                    registered.append(getattr(metric, hook))
                except AttributeError:
                    pass

            def closing_over_loop_variables_is_hard(registered=registered):
                def hook_implementation(*args, **kwargs):
                    for impl in registered:
                        impl(*args, **kwargs)

                return hook_implementation

            hook_implementation = closing_over_loop_variables_is_hard()

            hook_implementation.__name__ = hook
            setattr(self, hook, hook_implementation)

    def handle_start_of_simulation(self, benchmark_source):
        self._benchmark_source = benchmark_source

        self.start_of_simulation(
            self._ledger,
            self.emission_rate,
            self._trading_calendar,
            self._sessions,
            benchmark_source,
        )

    @property
    def portfolio(self):
        return self._ledger.portfolio

    @property
    def account(self):
        return self._ledger.account

    @property
    def positions(self):
        return self._ledger.position_tracker.positions

    def update_position(
        self,
        asset,
        amount=None,
        last_sale_price=None,
        last_sale_date=None,
        cost_basis=None,
    ):
        self._ledger.position_tracker.update_position(
            asset,
            amount,
            last_sale_price,
            last_sale_date,
            cost_basis,
        )

    def override_account_fields(self, **kwargs):
        self._ledger.override_account_fields(**kwargs)

    def process_transaction(self, transaction):
        self._ledger.process_transaction(transaction)

    def handle_splits(self, splits):
        self._ledger.process_splits(splits)

    def process_order(self, event):
        self._ledger.process_order(event)

    def process_commission(self, commission):
        self._ledger.process_commission(commission)

    def process_close_position(self, asset, dt, data_portal):
        self._ledger.close_position(asset, dt, data_portal)

    def capital_change(self, amount):
        self._ledger.capital_change(amount)

    def sync_last_sale_prices(self, dt, data_portal, handle_non_market_minutes=False):
        self._ledger.sync_last_sale_prices(
            dt,
            data_portal,
            handle_non_market_minutes=handle_non_market_minutes,
        )

    def handle_minute_close(self, dt, data_portal):
        """Handles the close of the given minute in minute emission.

        Parameters
        ----------
        dt : Timestamp
            The minute that is ending

        Returns
        -------
        A minute perf packet.
        """
        self.sync_last_sale_prices(dt, data_portal)

        packet = {
            "period_start": self._first_session,
            "period_end": self._last_session,
            "capital_base": self._capital_base,
            "minute_perf": {
                "period_open": self._market_open,
                "period_close": dt,
            },
            "cumulative_perf": {
                "period_open": self._first_session,
                "period_close": self._last_session,
            },
            "progress": self._progress(self),
            "cumulative_risk_metrics": {},
        }
        ledger = self._ledger
        ledger.end_of_bar(self._session_count)
        self.end_of_bar(
            packet,
            ledger,
            dt,
            self._session_count,
            data_portal,
        )
        return packet

    def handle_market_open(self, session_label, data_portal):
        """Handles the start of each session.

        Parameters
        ----------
        session_label : Timestamp
            The label of the session that is about to begin.
        data_portal : DataPortal
            The current data portal.
        """
        ledger = self._ledger
        ledger.start_of_session(session_label)

        adjustment_reader = data_portal.adjustment_reader
        if adjustment_reader is not None:
            # this is None when running with a dataframe source
            ledger.process_dividends(
                session_label,
                self._asset_finder,
                adjustment_reader,
            )

        self._current_session = session_label

        cal = self._trading_calendar
        self._market_open, self._market_close = self._execution_open_and_close(
            cal,
            session_label,
        )

        self.start_of_session(ledger, session_label, data_portal)

    def handle_market_close(self, dt, data_portal):
        """Handles the close of the given day.

        Parameters
        ----------
        dt : Timestamp
            The most recently completed simulation datetime.
        data_portal : DataPortal
            The current data portal.

        Returns
        -------
        A daily perf packet.
        """
        completed_session = self._current_session

        if self.emission_rate == "daily":
            # this method is called for both minutely and daily emissions, but
            # this chunk of code here only applies for daily emissions. (since
            # it's done every minute, elsewhere, for minutely emission).
            self.sync_last_sale_prices(dt, data_portal)

        session_ix = self._session_count
        # increment the day counter before we move markers forward.
        self._session_count += 1

        packet = {
            "period_start": self._first_session,
            "period_end": self._last_session,
            "capital_base": self._capital_base,
            "daily_perf": {
                "period_open": self._market_open,
                "period_close": dt,
            },
            "cumulative_perf": {
                "period_open": self._first_session,
                "period_close": self._last_session,
            },
            "progress": self._progress(self),
            "cumulative_risk_metrics": {},
        }
        ledger = self._ledger
        ledger.end_of_session(session_ix)
        self.end_of_session(
            packet,
            ledger,
            completed_session,
            session_ix,
            data_portal,
        )

        return packet

    def handle_simulation_end(self, data_portal):
        """When the simulation is complete, run the full period risk report
        and send it out on the results socket.
        """
        log.info(
            "Simulated %(days)s trading days\n first open: %(first)s\n last close: %(last)s",
            dict(
                days=self._session_count,
                first=self._trading_calendar.session_open(self._first_session),
                last=self._trading_calendar.session_close(self._last_session),
            ),
        )

        packet = {}
        self.end_of_simulation(
            packet,
            self._ledger,
            self._trading_calendar,
            self._sessions,
            data_portal,
            self._benchmark_source,
        )
        return packet
