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
import logbook

from zipline.algorithm import TradingAlgorithm
from zipline.gens.realtimeclock import RealtimeClock
from zipline.gens.tradesimulation import AlgorithmSimulator
from zipline.errors import OrderInBeforeTradingStart
from zipline.utils.api_support import (
    api_method,
    require_initialized,
    require_not_initialized,
    ZiplineAPI,
    disallowed_in_before_trading_start)

log = logbook.Logger("Live Trading")


class LiveAlgorithmExecutor(AlgorithmSimulator):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)


class LiveTradingAlgorithm(TradingAlgorithm):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

        self.live_trading = kwargs.pop('live_trading', False)
        self.broker = kwargs.pop('broker', None)

        log.info("Yippie, live!")

    def _create_clock(self):
        sim_clock = TradingAlgorithm._create_clock()

        return RealtimeClock(
            sim_clock.sessions,
            sim_clock.execution_opens,
            sim_clock.execution_closes,
            sim_clock.before_trading_start_minutes,
            sim_clock.minutely_emission,
            self.broker.time_skew
        )

    def _create_generator(self, sim_params):
        # Call the simulation trading algorithm for side-effects:
        # it creates the perf tracker
        _ = TradingAlgorithm._create_generator(self, sim_params)
        log.info("Live trading")
        self.trading_client = LiveAlgorithmExecutor(
            self,
            sim_params,
            self.data_portal,
            self._create_clock(),
            self._create_benchmark_source(),
            self.restrictions,
            universe_func=self._calculate_universe
        )

        return self.trading_client.transform()

    def updated_portfolio(self):
        return self.broker.portfolio

    def updated_account(self):
        return self.broker.account

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order(self,
              asset,
              amount,
              limit_price=None,
              stop_price=None,
              style=None):
        raise NotImplementedError()

    @api_method
    def batch_market_order(self, share_counts):
        raise NotImplementedError()

    def get_open_orders(self, asset=None):
        raise NotImplementedError()

    def get_order(self, order_id):
        raise NotImplementedError()

    def cancel_order(self, order_param):
        raise NotImplementedError()
