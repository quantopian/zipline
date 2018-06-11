#
# Copyright 2015 Quantopian, Inc.
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
from datetime import time

import logbook
from zipline.gens.sim_engine import MinuteSimulationClock
from zipline.gens.tradesimulation import AlgorithmSimulator
from zipline.utils.calendars.trading_calendar import days_at_time
from . import TradingAlgorithm

log = logbook.Logger("ZiplineLog")


class SimulatedTradingAlgorithm(TradingAlgorithm):
    """A class that represents a trading strategy and parameters to execute
    the strategy.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to ``initialize`` unless listed below.
    initialize : callable[context -> None], optional
        Function that is called at the start of the simulation to
        setup the initial context.
    handle_data : callable[(context, data) -> None], optional
        Function called on every bar. This is where most logic should be
        implemented.
    before_trading_start : callable[(context, data) -> None], optional
        Function that is called before any bars have been processed each
        day.
    analyze : callable[(context, DataFrame) -> None], optional
        Function that is called at the end of the backtest. This is passed
        the context and the performance results for the backtest.
    script : str, optional
        Algoscript that contains the definitions for the four algorithm
        lifecycle functions and any supporting code.
    namespace : dict, optional
        The namespace to execute the algoscript in. By default this is an
        empty namespace that will include only python built ins.
    algo_filename : str, optional
        The filename for the algoscript. This will be used in exception
        tracebacks. default: '<string>'.
    data_frequency : {'daily', 'minute'}, optional
        The duration of the bars.
    instant_fill : bool, optional
        Whether to fill orders immediately or on next bar. default: False
    equities_metadata : dict or DataFrame or file-like object, optional
        If dict is provided, it must have the following structure:
        * keys are the identifiers
        * values are dicts containing the metadata, with the metadata
          field name as the key
        If pandas.DataFrame is provided, it must have the
        following structure:
        * column names must be the metadata fields
        * index must be the different asset identifiers
        * array contents should be the metadata value
        If an object with a ``read`` method is provided, ``read`` must
        return rows containing at least one of 'sid' or 'symbol' along
        with the other metadata fields.
    futures_metadata : dict or DataFrame or file-like object, optional
        The same layout as ``equities_metadata`` except that it is used
        for futures information.
    identifiers : list, optional
        Any asset identifiers that are not provided in the
        equities_metadata, but will be traded by this TradingAlgorithm.
    get_pipeline_loader : callable[BoundColumn -> PipelineLoader], optional
        The function that maps pipeline columns to their loaders.
    create_event_context : callable[BarData -> context manager], optional
        A function used to create a context mananger that wraps the
        execution of all events that are scheduled for a bar.
        This function will be passed the data for the bar and should
        return the actual context manager that will be entered.
    history_container_class : type, optional
        The type of history container to use. default: HistoryContainer
    platform : str, optional
        The platform the simulation is running on. This can be queried for
        in the simulation with ``get_environment``. This allows algorithms
        to conditionally execute code based on platform it is running on.
        default: 'zipline'
    adjustment_reader : AdjustmentReader
        The interface to the adjustments.
    """

    def __init__(self, *args, **kwargs):
        """Initialize sids and other state variables.

        :Arguments:
        :Optional:
            initialize : function
                Function that is called with a single
                argument at the begninning of the simulation.
            handle_data : function
                Function that is called with 2 arguments
                (context and data) on every bar.
            script : str
                Algoscript that contains initialize and
                handle_data function definition.
            data_frequency : {'daily', 'minute'}
               The duration of the bars.
            capital_base : float <default: 1.0e5>
               How much capital to start with.
            asset_finder : An AssetFinder object
                A new AssetFinder object to be used in this TradingEnvironment
            equities_metadata : can be either:
                            - dict
                            - pandas.DataFrame
                            - object with 'read' property
                If dict is provided, it must have the following structure:
                * keys are the identifiers
                * values are dicts containing the metadata, with the metadata
                  field name as the key
                If pandas.DataFrame is provided, it must have the
                following structure:
                * column names must be the metadata fields
                * index must be the different asset identifiers
                * array contents should be the metadata value
                If an object with a 'read' property is provided, 'read' must
                return rows containing at least one of 'sid' or 'symbol' along
                with the other metadata fields.
            identifiers : List
                Any asset identifiers that are not provided in the
                equities_metadata, but will be traded by this TradingAlgorithm
        """

        super(SimulatedTradingAlgorithm, self).__init__(*args, **kwargs)

    def _create_clock(self):
        """
        If the clock property is not set, then create one based on frequency.
        """
        trading_o_and_c = self.trading_calendar.schedule.ix[
            self.sim_params.sessions]
        market_closes = trading_o_and_c['market_close']
        minutely_emission = False

        if self.sim_params.data_frequency == 'minute':
            market_opens = trading_o_and_c['market_open']

            minutely_emission = self.sim_params.emission_rate == "minute"
        else:
            # in daily mode, we want to have one bar per session, timestamped
            # as the last minute of the session.
            market_opens = market_closes

        # The calendar's execution times are the minutes over which we actually
        # want to run the clock. Typically the execution times simply adhere to
        # the market open and close times. In the case of the futures calendar,
        # for example, we only want to simulate over a subset of the full 24
        # hour calendar, so the execution times dictate a market open time of
        # 6:31am US/Eastern and a close of 5:00pm US/Eastern.
        execution_opens = \
            self.trading_calendar.execution_time_from_open(market_opens)
        execution_closes = \
            self.trading_calendar.execution_time_from_close(market_closes)

        # FIXME generalize these values
        before_trading_start_minutes = days_at_time(
            self.sim_params.sessions,
            time(8, 45),
            "US/Eastern"
        )

        return MinuteSimulationClock(
            self.sim_params.sessions,
            execution_opens,
            execution_closes,
            before_trading_start_minutes,
            minute_emission=minutely_emission,
        )

    def _create_generator(self, sim_params):
        if sim_params is not None:
            self.sim_params = sim_params

        self.metrics_tracker = metrics_tracker = self._create_metrics_tracker()

        # Set the dt initially to the period start by forcing it to change.
        self.on_dt_changed(self.sim_params.start_session)

        if not self.initialized:
            self.initialize(*self.initialize_args, **self.initialize_kwargs)
            self.initialized = True

        benchmark_source = self._create_benchmark_source()

        self.trading_client = AlgorithmSimulator(
            self,
            sim_params,
            self.data_portal,
            self._create_clock(),
            benchmark_source,
            self.restrictions,
            universe_func=self._calculate_universe
        )

        metrics_tracker.handle_start_of_simulation(benchmark_source)
        return self.trading_client.transform()
