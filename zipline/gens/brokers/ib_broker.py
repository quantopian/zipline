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

import sys
from collections import namedtuple, defaultdict
from time import sleep
from math import fabs

from six import itervalues
import pandas as pd
import numpy as np

from zipline.gens.brokers.broker import Broker
from zipline.finance.order import (Order as ZPOrder,
                                   ORDER_STATUS as ZP_ORDER_STATUS)
from zipline.finance.execution import (MarketOrder,
                                       LimitOrder,
                                       StopOrder,
                                       StopLimitOrder)
import zipline.protocol as zp
from zipline.api import symbol as symbol_lookup
from zipline.errors import SymbolNotFound

from ib.ext.EClientSocket import EClientSocket
from ib.ext.EWrapper import EWrapper
from ib.ext.Contract import Contract
from ib.ext.Order import Order
from ib.ext.EClientErrors import EClientErrors

from logbook import Logger

if sys.version_info > (3,):
    long = int

log = Logger('IB Broker')

Position = namedtuple('Position', ['contract', 'position', 'market_price',
                                   'market_value', 'average_cost',
                                   'unrealized_pnl', 'realized_pnl',
                                   'account_name'])


def log_message(message, mapping):
    try:
        del(mapping['self'])
    except (KeyError, ):
        pass
    items = list(mapping.items())
    items.sort()
    log.debug(('### %s' % (message, )))
    for k, v in items:
        log.debug(('    %s:%s' % (k, v)))


class TWSConnection(EClientSocket, EWrapper):
    def __init__(self, tws_uri, order_update_callback):
        EWrapper.__init__(self)
        EClientSocket.__init__(self, anyWrapper=self)

        self.tws_uri = tws_uri
        host, port, client_id = self.tws_uri.split(':')
        self._order_update_callback = order_update_callback

        self._next_ticker_id = 0
        self._next_order_id = None
        self.managed_accounts = None
        self.symbol_to_ticker_id = {}
        self.ticker_id_to_symbol = {}
        self.last_tick = defaultdict(dict)
        self.bars = {}
        # accounts structure: accounts[account_id][currency][value]
        self.accounts = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: np.NaN)))
        self.accounts_download_complete = False
        self.positions = {}
        self.portfolio = {}
        self.orders = {}
        self.time_skew = None

        log.info("Connecting: {}:{}:{}".format(host, int(port),
                                               int(client_id)))
        self.eConnect(host, int(port), int(client_id))
        while self.notConnected():
            sleep(0.1)

        self._download_account_details()
        log.info("Managed accounts: {}".format(self.managed_accounts))

        self.reqCurrentTime()
        self.reqIds(1)

        while self.time_skew is None or self._next_order_id is None:
            sleep(0.1)

        log.info("Local-Broker Time Skew: {}".format(self.time_skew))

    def _download_account_details(self):
        self.reqManagedAccts()
        while self.managed_accounts is None:
            sleep(0.1)

        for account in self.managed_accounts:
            self.reqAccountUpdates(subscribe=True, acctCode=account)
        while self.accounts_download_complete is False:
            sleep(0.1)

    @property
    def next_ticker_id(self):
        ticker_id = self._next_ticker_id
        self._next_ticker_id += 1
        return ticker_id

    @property
    def next_order_id(self):
        order_id = self._next_order_id
        self._next_order_id += 1
        return order_id

    def subscribe_to_market_data(self,
                                 symbol,
                                 sec_type='STK',
                                 exchange='SMART',
                                 currency='USD'):
        if symbol in self.symbol_to_ticker_id:
            # Already subscribed to market data
            return

        contract = Contract()
        contract.m_symbol = symbol
        contract.m_secType = sec_type
        contract.m_exchange = exchange
        contract.m_currency = currency

        ticker_id = self.next_ticker_id

        self.symbol_to_ticker_id[symbol] = ticker_id
        self.ticker_id_to_symbol[ticker_id] = symbol

        tick_list = "233"  # RTVolume, return tick_type == 48
        self.reqMktData(ticker_id, contract, tick_list, False)

    def _process_tick(self, ticker_id, tick_type, value):
        try:
            symbol = self.ticker_id_to_symbol[ticker_id]
        except KeyError:
            log.error("Tick {} for id={} is not registered".format(tick_type,
                                                                   ticker_id))
            return
        if tick_type == 48:
            # RT Volume Bar. Format:
            # Last trade price; Last trade size;Last trade time;Total volume;\
            # VWAP;Single trade flag
            # e.g.: 701.28;1;1348075471534;67854;701.46918464;true
            (last_trade_price, last_trade_size, last_trade_time, total_volume,
             vwap, single_trade_flag) = value.split(';')

            # Ignore this update if last_trade_price is empty:
            # tickString: tickerId=0 tickType=48/RTVolume ;0;1469805548873;\
            # 240304;216.648653;true
            if len(last_trade_price) == 0:
                return

            last_trade_dt = pd.to_datetime(float(last_trade_time), unit='ms',
                                           utc=True)

            self._add_bar(symbol, float(last_trade_price),
                          int(last_trade_size), last_trade_dt,
                          int(total_volume), float(vwap),
                          single_trade_flag)

    def _add_bar(self, symbol, last_trade_price, last_trade_size,
                 last_trade_time, total_volume, vwap, single_trade_flag):
        bar = pd.DataFrame(index=pd.DatetimeIndex([last_trade_time]),
                           data={'last_trade_price': last_trade_price,
                                 'last_trade_size': last_trade_size,
                                 'total_volume': total_volume,
                                 'vwap': vwap,
                                 'single_trade_flag': single_trade_flag})

        if symbol not in self.bars:
            self.bars[symbol] = bar
        else:
            self.bars[symbol] = self.bars[symbol].append(bar)

    def tickPrice(self, ticker_id, field, price, can_auto_execute):
        self._process_tick(ticker_id, tick_type=field, value=price)

    def tickSize(self, ticker_id, field, size):
        self._process_tick(ticker_id, tick_type=field, value=size)

    def tickOptionComputation(self,
                              ticker_id, field, implied_vol, delta, opt_price,
                              pv_dividend, gamma, vega, theta, und_price):
        log_message('tickOptionComputation', vars())

    def tickGeneric(self, ticker_id, tick_type, value):
        self._process_tick(ticker_id, tick_type=tick_type, value=value)

    def tickString(self, ticker_id, tick_type, value):
        self._process_tick(ticker_id, tick_type=tick_type, value=value)

    def tickEFP(self, ticker_id, tick_type, basis_points,
                formatted_basis_points, implied_future, hold_days,
                future_expiry, dividend_impact, dividends_to_expiry):
        log_message('tickEFP', vars())

    def orderStatus(self, order_id, status, filled, remaining, avg_fill_price,
                    perm_id, parent_id, last_fill_price, client_id, why_held):
        log_message('orderStatus', vars())

        self._order_update_callback(order_id, status, int(filled))

    def openOrder(self, order_id, contract, order, state):
        log_message('openOrder', vars())

    def openOrderEnd(self):
        log_message('openOrderEnd', vars())

    def updateAccountValue(self, key, value, currency, account_name):
        self.accounts[account_name][currency][key] = value

    def updatePortfolio(self,
                        contract,
                        position,
                        market_price,
                        market_value,
                        average_cost,
                        unrealized_pnl,
                        realized_pnl,
                        account_name):
        symbol = contract.m_symbol

        position = Position(contract=contract,
                            position=position,
                            market_price=market_price,
                            market_value=market_value,
                            average_cost=average_cost,
                            unrealized_pnl=unrealized_pnl,
                            realized_pnl=realized_pnl,
                            account_name=account_name)

        self.positions[symbol] = position

    def updateAccountTime(self, time_stamp):
        pass

    def accountDownloadEnd(self, account_name):
        self.accounts_download_complete = True

    def nextValidId(self, order_id):
        self._next_order_id = order_id

    def contractDetails(self, req_id, contract_details):
        log_message('contractDetails', vars())

    def contractDetailsEnd(self, req_id):
        log_message('contractDetailsEnd', vars())

    def bondContractDetails(self, req_id, contract_details):
        log_message('bondContractDetails', vars())

    def execDetails(self, req_id, contract, execution):
        log_message('execDetails', vars())

    def execDetailsEnd(self, req_id):
        log_message('execDetailsEnd', vars())

    def connectionClosed(self):
        log_message('connectionClosed', {})

    def error(self, id_=None, error_code=None, error_msg=None):
        if isinstance(error_code, int):
            if error_code < 1000:
                log.error("[{}] {} ({})".format(error_code, error_msg, id_))
            else:
                log.info("[{}] {}".format(error_code, error_msg, id_))
        elif isinstance(error_code, EClientErrors.CodeMsgPair):
            log.error("[{}] {}".format(error_code.code(),
                                       error_code.msg(),
                                       id_))
        else:
            log.error("[{}] {} ({})".format(error_code, error_msg, id_))

    def updateMktDepth(self, ticker_id, position, operation, side, price,
                       size):
        log_message('updateMktDepth', vars())

    def updateMktDepthL2(self, ticker_id, position, market_maker, operation,
                         side, price, size):
        log_message('updateMktDepthL2', vars())

    def updateNewsBulletin(self, msg_id, msg_type, message, orig_exchange):
        log_message('updateNewsBulletin', vars())

    def managedAccounts(self, accounts_list):
        self.managed_accounts = accounts_list.split(',')

    def receiveFA(self, fa_data_type, xml):
        log_message('receiveFA', vars())

    def historicalData(self, req_id, date, open_, high, low, close, volume,
                       count, wap, has_gaps):
        log_message('historicalData', vars())

    def scannerParameters(self, xml):
        log_message('scannerParameters', vars())

    def scannerData(self, req_id, rank, contract_details, distance, benchmark,
                    projection, legs_str):
        log_message('scannerData', vars())

    def commissionReport(self, commission_report):
        log_message('commissionReport', vars())

    def currentTime(self, time):
        self.time_skew = (pd.to_datetime('now', utc=True) -
                          pd.to_datetime(long(time), unit='s', utc=True))

    def deltaNeutralValidation(self, req_id, under_comp):
        log_message('deltaNeutralValidation', vars())

    def fundamentalData(self, req_id, data):
        log_message('fundamentalData', vars())

    def marketDataType(self, req_id, market_data_type):
        log_message('marketDataType', vars())

    def realtimeBar(self, req_id, time, open_, high, low, close, volume, wap,
                    count):
        log_message('realtimeBar', vars())

    def scannerDataEnd(self, req_id):
        log_message('scannerDataEnd', vars())

    def tickSnapshotEnd(self, req_id):
        log_message('tickSnapshotEnd', vars())

    def position(self, account, contract, pos, avg_cost):
        log_message('position', vars())

    def positionEnd(self):
        log_message('positionEnd', vars())

    def accountSummary(self, req_id, account, tag, value, currency):
        log_message('accountSummary', vars())

    def accountSummaryEnd(self, req_id):
        log_message('accountSummaryEnd', vars())


class IBBroker(Broker):
    def __init__(self, tws_uri, account_id=None):
        self._tws_uri = tws_uri
        self.orders = {}

        self._tws = TWSConnection(tws_uri, self._order_update)
        self.account_id = (self._tws.managed_accounts[0] if account_id is None
                           else self._tws.managed_accounts[0])
        self.currency = 'USD'

        super(self.__class__, self).__init__()

    def subscribe_to_market_data(self, symbol):
        self._tws.subscribe_to_market_data(symbol)

    @property
    def positions(self):
        z_positions = zp.Positions()
        for symbol in self._tws.positions:
            ib_position = self._tws.positions[symbol]
            try:
                z_position = zp.Position(symbol_lookup(symbol))
            except SymbolNotFound:
                # The symbol might not have been ingested to the db therefore
                # it needs to be skipped.
                continue
            z_position.amount = int(ib_position.position)
            z_position.cost_basis = float(ib_position.market_price)
            z_position.last_sale_price = None  # TODO(tibor): Fill from state
            z_position.last_sale_date = None  # TODO(tibor): Fill from state
            z_positions[symbol_lookup(symbol)] = z_position

        return z_positions

    @property
    def portfolio(self):
        ib_account = self._tws.accounts[self.account_id][self.currency]

        z_portfolio = zp.Portfolio()
        z_portfolio.capital_used = None  # TODO(tibor)
        z_portfolio.starting_cash = None  # TODO(tibor): Fill from state
        z_portfolio.portfolio_value = float(ib_account['EquityWithLoanValue'])
        z_portfolio.pnl = (float(ib_account['RealizedPnL']) +
                           float(ib_account['UnrealizedPnL']))
        z_portfolio.returns = None  # TODO(tibor): pnl / total_at_start
        z_portfolio.cash = float(ib_account['TotalCashValue'])
        z_portfolio.start_date = None  # TODO(tibor)
        z_portfolio.positions = self.positions
        z_portfolio.positions_value = None  # TODO(tibor)
        z_portfolio.positions_exposure = None  # TODO(tibor)

        return z_portfolio

    @property
    def account(self):
        ib_account = self._tws.accounts[self.account_id][self.currency]

        z_account = zp.Account()

        z_account.settled_cash = None  # TODO(tibor)
        z_account.accrued_interest = None  # TODO(tibor)
        z_account.buying_power = float(ib_account['BuyingPower'])
        z_account.equity_with_loan = float(ib_account['EquityWithLoanValue'])
        z_account.total_positions_value = None  # TODO(tibor)
        z_account.total_positions_exposure = None  # TODO(tibor)
        z_account.regt_equity = float(ib_account['RegTEquity'])
        z_account.regt_margin = float(ib_account['RegTMargin'])
        z_account.initial_margin_requirement = float(
            ib_account['FullInitMarginReq'])
        z_account.maintenance_margin_requirement = float(
            ib_account['FullMaintMarginReq'])
        z_account.available_funds = None  # TODO(tibor)
        z_account.excess_liquidity = float(ib_account['ExcessLiquidity'])
        z_account.cushion = float(
            self._tws.accounts[self.account_id]['']['Cushion'])
        z_account.day_trades_remaining = float(
            self._tws.accounts[self.account_id]['']['DayTradesRemaining'])
        z_account.leverage = float(
            self._tws.accounts[self.account_id]['']['Leverage-S'])
        z_account.net_leverage = None  # TODO(tibor)
        z_account.net_liquidation = float(ib_account['NetLiquidation'])

        return z_account

    @property
    def time_skew(self):
        return self._tws.time_skew

    def order(self, asset, amount, limit_price, stop_price, style):
        is_buy = (amount > 0)
        zp_order = ZPOrder(
            dt=pd.to_datetime('now', utc=True),
            asset=asset,
            amount=amount,
            stop=style.get_stop_price(is_buy),
            limit=style.get_limit_price(is_buy))

        contract = Contract()
        contract.m_symbol = str(asset.symbol)
        contract.m_currency = self.currency
        contract.m_exchange = 'SMART'
        contract.m_secType = 'STK'

        order = Order()
        order.m_totalQuantity = int(fabs(amount))
        order.m_action = "BUY" if amount > 0 else "SELL"

        order.m_lmtPrice = 0
        order.m_auxPrice = 0

        if isinstance(style, MarketOrder):
            order.m_orderType = "MKT"
        elif isinstance(style, LimitOrder):
            order.m_orderType = "LMT"
            order.m_lmtPrice = limit_price
        elif isinstance(style, StopOrder):
            order.m_orderType = "STP"
            order.m_auxPrice = stop_price
        elif isinstance(style, StopLimitOrder):
            order.m_orderType = "STP LMT"
            order.m_auxPrice = stop_price
            order.m_lmtPrice = limit_price

        order.m_tif = "DAY"

        ib_order_id = self._tws.next_order_id
        zp_order.broker_order_id = ib_order_id
        self.orders[zp_order.id] = zp_order

        self._tws.placeOrder(ib_order_id, contract, order)

        return zp_order.id

    def get_open_orders(self, asset):
        if asset is None:
            assets = set([order.asset for order in itervalues(self.orders)
                          if order.open])
            return {
                asset: [order.to_api_obj() for order in itervalues(self.orders)
                        if order.asset == asset]
                for asset in assets
            }
        return [order.to_api_obj() for order in itervalues(self.orders)
                if order.asset == asset and order.open]

    def get_order(self, zp_order_id):
        return self.orders[zp_order_id].to_api_obj()

    def cancel_order(self, zp_order_id):
        ib_order_id = self.orders[zp_order_id].broker_order_id
        # ZPOrder cancellation will be done indirectly through _order_update
        self._tws.cancelOrder(ib_order_id)

    def _get_zp_order_id(self, ib_order_id):
        ib_order_ids = [e for e in self.orders
                        if self.orders[e].broker_order_id == ib_order_id]
        if len(ib_order_ids) == 0:
            return None
        elif len(ib_order_ids) == 1:
            return ib_order_ids[0]
        else:
            raise RuntimeError("More than one order found for id: %s" %
                               ib_order_id)

    def _order_update(self, ib_order_id, status, filled):
        # TWS can report orders which has not been registered in the current
        # session: If the app crashed and restarted (with the same client_id)
        # the orders fired prior to the crash is reported on restart.
        # Those order updates will be ignored by us.
        zp_order_id = self._get_zp_order_id(ib_order_id)
        if zp_order_id is None:
            return

        if status.lower() == 'submitted':
            self.orders[zp_order_id].status = ZP_ORDER_STATUS.OPEN
        elif status.lower() == 'cancelled':
            self.orders[zp_order_id].status = ZP_ORDER_STATUS.CANCELLED
        elif status.lower() == 'filled':
            self.orders[zp_order_id].status = ZP_ORDER_STATUS.FILLED

        self.orders[zp_order_id].filled = filled

        # TODO: Add commission if the order is executed

    def get_spot_value(self, assets, field, dt, data_frequency):
        symbol = str(assets.symbol)

        if symbol not in self._tws.bars:
            self._tws.subscribe_to_market_data(symbol)
            return pd.NaT if field == 'last_traded' else np.NaN

        bars = self._tws.bars[symbol]

        last_event_time = bars.index[-1]

        minute_start = (last_event_time - pd.Timedelta('1 min')) \
            .time()
        minute_end = last_event_time.time()

        if bars.empty:
            return pd.NaT if field == 'last_traded' else np.NaN
        else:
            if field == 'price':
                return bars.last_trade_price.iloc[-1]
            elif field == 'last_traded':
                return last_event_time or pd.NaT

            minute_df = bars.between_time(minute_start, minute_end,
                                          include_start=True, include_end=True)
            if minute_df.empty:
                return np.NaN
            else:
                if field == 'open':
                    return minute_df.last_trade_price.iloc[0]
                elif field == 'close':
                    return minute_df.last_trade_price.iloc[-1]
                elif field == 'high':
                    return minute_df.last_trade_price.max()
                elif field == 'low':
                    return minute_df.last_trade_price.min()
                elif field == 'volume':
                    return minute_df.last_trade_size.sum()
