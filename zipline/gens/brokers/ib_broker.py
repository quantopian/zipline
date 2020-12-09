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
from collections import namedtuple, defaultdict, OrderedDict
from time import sleep
from math import fabs

from six import iteritems, itervalues
import polling
import pandas as pd
import numpy as np

from zipline.gens.brokers.broker import Broker
from zipline.finance.order import (Order as ZPOrder,
                                   ORDER_STATUS as ZP_ORDER_STATUS)
from zipline.finance.execution import (MarketOrder,
                                       LimitOrder,
                                       StopOrder,
                                       StopLimitOrder)
from zipline.finance.transaction import Transaction
import zipline.protocol as zp
from zipline.protocol import MutableView
from zipline.api import symbol as symbol_lookup
from zipline.errors import SymbolNotFound

from ib.ext.EClientSocket import EClientSocket
from ib.ext.EWrapper import EWrapper
from ib.ext.Contract import Contract
from ib.ext.Order import Order
from ib.ext.ExecutionFilter import ExecutionFilter
from ib.ext.EClientErrors import EClientErrors

from logbook import Logger

if sys.version_info > (3,):
    long = int

log = Logger('IB Broker')

Position = namedtuple('Position', ['contract', 'position', 'market_price',
                                   'market_value', 'average_cost',
                                   'unrealized_pnl', 'realized_pnl',
                                   'account_name'])

_max_wait_subscribe = 10  # how many cycles to wait
_connection_timeout = 15  # Seconds
_poll_frequency = 0.1

symbol_to_exchange = defaultdict(lambda: 'SMART')
symbol_to_exchange['VIX'] = 'CBOE'
symbol_to_exchange['SPX'] = 'CBOE'
symbol_to_exchange['VIX3M'] = 'CBOE'
symbol_to_exchange['VXST'] = 'CBOE'
symbol_to_exchange['VXMT'] = 'CBOE'
symbol_to_exchange['GVZ'] = 'CBOE'
symbol_to_exchange['GLD'] = 'ARCA'
symbol_to_exchange['GDX'] = 'ARCA'
symbol_to_exchange['GPRO'] = 'SMART/NASDAQ'
symbol_to_exchange['MSFT'] = 'SMART/NASDAQ'
symbol_to_exchange['CSCO'] = 'SMART/NASDAQ'

symbol_to_sec_type = defaultdict(lambda: 'STK')
symbol_to_sec_type['VIX'] = 'IND'
symbol_to_sec_type['VIX3M'] = 'IND'
symbol_to_sec_type['VXST'] = 'IND'
symbol_to_sec_type['VXMT'] = 'IND'
symbol_to_sec_type['GVZ'] = 'IND'
symbol_to_sec_type['SPX'] = 'IND'


def log_message(message, mapping):
    try:
        del (mapping['self'])
    except (KeyError,):
        pass
    items = list(mapping.items())
    items.sort()
    log.debug(('### %s' % (message,)))
    for k, v in items:
        log.debug(('    %s:%s' % (k, v)))


def _method_params_to_dict(args):
    return {k: v
            for k, v in iteritems(args)
            if k != 'self'}


class TWSConnection(EClientSocket, EWrapper):
    def __init__(self, tws_uri):
        """
        :param tws_uri: host:listening_port:client_id
                        - host ip of running tws or ibgw
                        - port, default for tws 7496 and for ibgw 4002
                        - your client id, could be any number as long as it's not already used
        """
        EWrapper.__init__(self)
        EClientSocket.__init__(self, anyWrapper=self)

        self.tws_uri = tws_uri
        host, port, client_id = self.tws_uri.split(':')
        self._host = host
        self._port = int(port)
        self.client_id = int(client_id)

        self._next_ticker_id = 0
        self._next_request_id = 0
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
        self.open_orders = {}
        self.order_statuses = {}
        self.executions = defaultdict(OrderedDict)
        self.commissions = defaultdict(OrderedDict)
        self._execution_to_order_id = {}
        self.time_skew = None
        self.unrecoverable_error = False

        self.connect()

    def connect(self):
        log.info("Connecting: {}:{}:{}".format(self._host, self._port,
                                               self.client_id))
        self.eConnect(self._host, self._port, self.client_id)
        timeout = _connection_timeout
        while timeout > 0 and not self.isConnected():
            sleep(_poll_frequency)
            timeout -= _poll_frequency
        else:
            if not self.isConnected():
                raise SystemError("Connection timeout during TWS connection!")

        self._download_account_details()
        log.info("Managed accounts: {}".format(self.managed_accounts))

        self.reqCurrentTime()
        self.reqIds(1)

        while self.time_skew is None or self._next_order_id is None:
            sleep(_poll_frequency)

        log.info("Local-Broker Time Skew: {}".format(self.time_skew))

    def _download_account_details(self):
        exec_filter = ExecutionFilter()
        exec_filter.m_clientId = self.client_id
        self.reqExecutions(self.next_request_id, exec_filter)

        self.reqManagedAccts()
        while self.managed_accounts is None:
            sleep(_poll_frequency)

        for account in self.managed_accounts:
            self.reqAccountUpdates(subscribe=True, acctCode=account)
        while self.accounts_download_complete is False:
            sleep(_poll_frequency)

    @property
    def next_ticker_id(self):
        ticker_id = self._next_ticker_id
        self._next_ticker_id += 1
        return ticker_id

    @property
    def next_request_id(self):
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

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
        contract.m_secType = symbol_to_sec_type[symbol]
        contract.m_exchange = symbol_to_exchange[symbol]
        contract.m_currency = currency
        ticker_id = self.next_ticker_id

        self.symbol_to_ticker_id[symbol] = ticker_id
        self.ticker_id_to_symbol[ticker_id] = symbol

        # INDEX tickers cannot be requested with market data. The data can,
        # however, be requested with realtimeBars. This change will make
        # sure we can request data from INDEX tickers like SPX, VIX, etc.
        if contract.m_secType == 'IND':
            self.reqRealTimeBars(ticker_id, contract, 60, 'TRADES', True)
        else:
            tick_list = "233"  # RTVolume, return tick_type == 48
            self.reqMktData(ticker_id, contract, tick_list, False)
            sleep(11)

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
                           data={'last_trade_price':  last_trade_price,
                                 'last_trade_size':   last_trade_size,
                                 'total_volume':      total_volume,
                                 'vwap':              vwap,
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

    def orderStatus(self, order_id, status, filled, remaining, avg_fill_price,
                    perm_id, parent_id, last_fill_price, client_id, why_held):
        self.order_statuses[order_id] = _method_params_to_dict(vars())

        log.debug(
            "Order-{order_id} {status}: "
            "filled={filled} remaining={remaining} "
            "avg_fill_price={avg_fill_price} "
            "last_fill_price={last_fill_price} ".format(
                order_id=order_id,
                status=self.order_statuses[order_id]['status'],
                filled=self.order_statuses[order_id]['filled'],
                remaining=self.order_statuses[order_id]['remaining'],
                avg_fill_price=self.order_statuses[order_id]['avg_fill_price'],
                last_fill_price=self.order_statuses[order_id]['last_fill_price']))

    def openOrder(self, order_id, contract, order, state):
        self.open_orders[order_id] = _method_params_to_dict(vars())

        log.debug(
            "Order-{order_id} {status}: "
            "{order_action} {order_count} {symbol} with {order_type} order. "
            "limit_price={limit_price} stop_price={stop_price}".format(
                order_id=order_id,
                status=state.m_status,
                order_action=order.m_action,
                order_count=order.m_totalQuantity,
                symbol=contract.m_symbol,
                order_type=order.m_orderType,
                limit_price=order.m_lmtPrice,
                stop_price=order.m_auxPrice))

    def openOrderEnd(self):
        pass

    def execDetails(self, req_id, contract, exec_detail):
        order_id, exec_id = exec_detail.m_orderId, exec_detail.m_execId
        self.executions[order_id][exec_id] = _method_params_to_dict(vars())
        self._execution_to_order_id[exec_id] = order_id

        log.info(
            "Order-{order_id} executed @ {exec_time}: "
            "{symbol} current: {shares} @ ${price} "
            "total: {cum_qty} @ ${avg_price} "
            "exec_id: {exec_id} by client-{client_id}".format(
                order_id=order_id, exec_id=exec_id,
                exec_time=pd.to_datetime(exec_detail.m_time),
                symbol=contract.m_symbol,
                shares=exec_detail.m_shares,
                price=exec_detail.m_price,
                cum_qty=exec_detail.m_cumQty,
                avg_price=exec_detail.m_avgPrice,
                client_id=exec_detail.m_clientId))

    def execDetailsEnd(self, req_id):
        log.debug(
            "Execution details completed for request {req_id}".format(
                req_id=req_id))

    def commissionReport(self, commission_report):
        exec_id = commission_report.m_execId
        order_id = self._execution_to_order_id[commission_report.m_execId]
        self.commissions[order_id][exec_id] = commission_report

        log.debug(
            "Order-{order_id} report: "
            "realized_pnl: ${realized_pnl} "
            "commission: ${commission} yield: {yield_} "
            "exec_id: {exec_id}".format(
                order_id=order_id,
                exec_id=commission_report.m_execId,
                realized_pnl=commission_report.m_realizedPNL
                if commission_report.m_realizedPNL != sys.float_info.max
                else 0,
                commission=commission_report.m_commission,
                yield_=commission_report.m_yield
                if commission_report.m_yield != sys.float_info.max
                else 0)
        )

    def connectionClosed(self):
        self.unrecoverable_error = True
        log.error("IB Connection closed")

    def error(self, id_=None, error_code=None, error_msg=None):
        if isinstance(id_, Exception):
            # XXX: for an unknown reason 'log' is None in this branch,
            # therefore it needs to be instantiated before use
            global log
            if not log:
                log = Logger('IB Broker')
            log.exception(id_)

        if isinstance(error_code, EClientErrors.CodeMsgPair):
            error_msg = error_code.msg()
            error_code = error_code.code()

        if isinstance(error_code, int):
            if error_code in (502, 503, 326):
                # 502: Couldn't connect to TWS.
                # 503: The TWS is out of date and must be upgraded.
                # 326: Unable connect as the client id is already in use.
                self.unrecoverable_error = True

            if error_code < 1000:
                log.error("[{}] {} ({})".format(error_code, error_msg, id_))
            else:
                log.info("[{}] {} ({})".format(error_code, error_msg, id_))
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
        value = (";".join([str(close), str(count), str(time), str(volume),
                           str(wap), "true"]))
        self._process_tick(req_id, tick_type=48, value=value)

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
        """
        :param tws_uri: host:listening_port:client_id
                        - host ip of running tws or ibgw
                        - port, default for tws 7496 and for ibgw 4002
                        - your client id, could be any number as long as it's not already used
        """
        self._tws_uri = tws_uri
        self._orders = {}
        self._transactions = {}

        self._tws = TWSConnection(tws_uri)
        self.account_id = (self._tws.managed_accounts[0] if account_id is None
                           else account_id)
        self.currency = 'USD'

        self._subscribed_assets = []

        super(self.__class__, self).__init__()

    @property
    def subscribed_assets(self):
        return self._subscribed_assets

    def subscribe_to_market_data(self, asset):
        if asset not in self.subscribed_assets:
            log.info("Subscribing to market data for {}".format(
                asset))

            # remove str() cast to have a fun debugging journey
            self._tws.subscribe_to_market_data(str(asset.symbol))
            self._subscribed_assets.append(asset)
            try:
                polling.poll(
                    lambda: asset.symbol in self._tws.bars,
                    timeout=_max_wait_subscribe,
                    step=_poll_frequency)
            except polling.TimeoutException as te:
                log.warning('!!!WARNING: I did not manage to subscribe to %s ' % str(asset.symbol))
            else:
                log.debug("Subscription completed")

    @property
    def positions(self):
        self._get_positions_from_broker()
        return self.metrics_tracker.positions

    def _get_positions_from_broker(self):
        """
        get the positions from the broker and update zipline objects ( the ledger )
        should be used once at startup and once every time we want to refresh the positions array
        """
        cur_pos_in_tracker = self.metrics_tracker.positions
        for symbol in self._tws.positions:
            ib_position = self._tws.positions[symbol]
            try:
                z_position = zp.Position(zp.InnerPosition(symbol_lookup(symbol)))
                editable_position = MutableView(z_position)
            except SymbolNotFound:
                # The symbol might not have been ingested to the db therefore
                # it needs to be skipped.
                log.warning('Wanted to subscribe to %s, but this asset is probably not ingested' % symbol)
                continue
            editable_position._underlying_position.amount = int(ib_position.position)
            editable_position._underlying_position.cost_basis = float(ib_position.average_cost)
            # Check if symbol exists in bars df
            if symbol in self._tws.bars:
                editable_position._underlying_position.last_sale_price = \
                    float(self._tws.bars[symbol].last_trade_price.iloc[-1])
                editable_position._underlying_position.last_sale_date = \
                    self._tws.bars[symbol].index.values[-1]
            else:
                # editable_position._underlying_position.last_sale_price = None  # this cannot be set to None. only numbers.
                editable_position._underlying_position.last_sale_date = None
            self.metrics_tracker.update_position(z_position.asset,
                                                 amount=z_position.amount,
                                                 last_sale_price=z_position.last_sale_price,
                                                 last_sale_date=z_position.last_sale_date,
                                                 cost_basis=z_position.cost_basis)
        for asset in cur_pos_in_tracker:
            if asset.symbol not in self._tws.positions:
                # deleting object from the metrcs_tracker as its not in the portfolio
                self.metrics_tracker.update_position(asset,
                                                     amount=0)
        # for some reason, the metrics tracker has self.positions AND self.portfolio.positions. let's make sure
        # these objects are consistent
        # (self.portfolio.positions is self.metrics_tracker._ledger._portfolio.positions)
        # (self.metrics_tracker.positions is self.metrics_tracker._ledger.position_tracker.positions)
        self.metrics_tracker._ledger._portfolio.positions = self.metrics_tracker.positions


    @property
    def portfolio(self):
        positions = self.positions

        return self.metrics_tracker.portfolio

    def get_account_from_broker(self):
        ib_account = self._tws.accounts[self.account_id][self.currency]
        return ib_account

    
    def set_metrics_tracker(self, metrics_tracker):
        self.metrics_tracker = metrics_tracker

    @property
    def account(self):
        ib_account = self._tws.accounts[self.account_id][self.currency]

        self.metrics_tracker.override_account_fields(
            settled_cash=float(ib_account['CashBalance']),
            accrued_interest=float(ib_account['AccruedCash']),
            buying_power=float(ib_account['BuyingPower']),
            equity_with_loan=float(ib_account['EquityWithLoanValue']),
            total_positions_value=float(ib_account['StockMarketValue']),
            total_positions_exposure=float(
                (float(ib_account['StockMarketValue']) /
                 (float(ib_account['StockMarketValue']) +
                  float(ib_account['TotalCashValue'])))),
            regt_equity=float(ib_account['RegTEquity']),
            regt_margin=float(ib_account['RegTMargin']),
            initial_margin_requirement=float(
                ib_account['FullInitMarginReq']),
            maintenance_margin_requirement=float(
                ib_account['FullMaintMarginReq']),
            available_funds=float(ib_account['AvailableFunds']),
            excess_liquidity=float(ib_account['ExcessLiquidity']),
            cushion=float(
                self._tws.accounts[self.account_id]['']['Cushion']),
            day_trades_remaining=float(
                self._tws.accounts[self.account_id]['']['DayTradesRemaining']),
            leverage=float(
                self._tws.accounts[self.account_id]['']['Leverage-S']),
            net_leverage=(
                    float(ib_account['StockMarketValue']) /
                    (float(ib_account['TotalCashValue']) +
                     float(ib_account['StockMarketValue']))),
            net_liquidation=float(ib_account['NetLiquidation'])
        )

        return self.metrics_tracker.account

    @property
    def time_skew(self):
        return self._tws.time_skew

    def is_alive(self):
        return not self._tws.unrecoverable_error

    @staticmethod
    def _safe_symbol_lookup(symbol):
        try:
            return symbol_lookup(symbol)
        except SymbolNotFound:
            return None

    _zl_order_ref_magic = '!ZL'

    @classmethod
    def _create_order_ref(cls, ib_order, dt=pd.to_datetime('now', utc=True)):
        order_type = ib_order.m_orderType.replace(' ', '_')
        return \
            "A:{action} Q:{qty} T:{order_type} " \
            "L:{limit_price} S:{stop_price} D:{date} {magic}".format(
                action=ib_order.m_action,
                qty=ib_order.m_totalQuantity,
                order_type=order_type,
                limit_price=ib_order.m_lmtPrice,
                stop_price=ib_order.m_auxPrice,
                date=int(dt.value / 1e9),
                magic=cls._zl_order_ref_magic)

    @classmethod
    def _parse_order_ref(cls, ib_order_ref):
        if not ib_order_ref or \
                not ib_order_ref.endswith(cls._zl_order_ref_magic):
            return None

        try:
            action, qty, order_type, limit_price, stop_price, dt, _ = \
                ib_order_ref.split(' ')

            if not all(
                    [action.startswith('A:'),
                     qty.startswith('Q:'),
                     order_type.startswith('T:'),
                     limit_price.startswith('L:'),
                     stop_price.startswith('S:'),
                     dt.startswith('D:')]):
                return None

            return {
                'action':      action[2:],
                'qty':         int(qty[2:]),
                'order_type':  order_type[2:].replace('_', ' '),
                'limit_price': float(limit_price[2:]),
                'stop_price':  float(stop_price[2:]),
                'dt':          pd.to_datetime(dt[2:], unit='s', utc=True)}

        except ValueError:
            log.warning("Error parsing order metadata: {}".format(
                ib_order_ref))
            return None

    def order(self, asset, amount, style):
        contract = Contract()
        contract.m_symbol = str(asset.symbol)
        contract.m_currency = self.currency
        contract.m_exchange = symbol_to_exchange[str(asset.symbol)]
        contract.m_secType = symbol_to_sec_type[str(asset.symbol)]

        order = Order()
        order.m_totalQuantity = int(fabs(amount))
        order.m_action = "BUY" if amount > 0 else "SELL"

        is_buy = (amount > 0)
        order.m_lmtPrice = style.get_limit_price(is_buy) or 0
        order.m_auxPrice = style.get_stop_price(is_buy) or 0

        if isinstance(style, MarketOrder):
            order.m_orderType = "MKT"
        elif isinstance(style, LimitOrder):
            order.m_orderType = "LMT"
        elif isinstance(style, StopOrder):
            order.m_orderType = "STP"
        elif isinstance(style, StopLimitOrder):
            order.m_orderType = "STP LMT"

        # TODO: Support GTC orders both here and at blotter_live
        order.m_tif = "DAY"
        order.m_orderRef = self._create_order_ref(order)

        ib_order_id = self._tws.next_order_id
        zp_order = self._get_or_create_zp_order(ib_order_id, order, contract)

        log.info(
            "Placing order-{order_id}: "
            "{action} {qty} {symbol} with {order_type} order. "
            "limit_price={limit_price} stop_price={stop_price} {tif}".format(
                order_id=ib_order_id,
                action=order.m_action,
                qty=order.m_totalQuantity,
                symbol=contract.m_symbol,
                order_type=order.m_orderType,
                limit_price=order.m_lmtPrice,
                stop_price=order.m_auxPrice,
                tif=order.m_tif
            ))

        self._tws.placeOrder(ib_order_id, contract, order)

        return zp_order

    @property
    def orders(self):
        self._update_orders()
        return self._orders

    def _ib_to_zp_order_id(self, ib_order_id):
        return "IB-{date}-{account_id}-{client_id}-{order_id}".format(
            date=str(pd.to_datetime('today').date()),
            account_id=self.account_id,
            client_id=self._tws.client_id,
            order_id=ib_order_id)

    @staticmethod
    def _action_qty_to_amount(action, qty):
        return qty if action == 'BUY' else -1 * qty

    def _get_or_create_zp_order(self, ib_order_id,
                                ib_order=None, ib_contract=None):
        zp_order_id = self._ib_to_zp_order_id(ib_order_id)
        if zp_order_id in self._orders:
            return self._orders[zp_order_id]

        # Try to reconstruct the order from the given information:
        # open order state and execution state
        symbol, order_details = None, None

        if ib_order and ib_contract:
            symbol = ib_contract.m_symbol
            order_details = self._parse_order_ref(ib_order.m_orderRef)

        if not order_details and ib_order_id in self._tws.open_orders:
            open_order = self._tws.open_orders[ib_order_id]
            symbol = open_order['contract'].m_symbol
            order_details = self._parse_order_ref(
                open_order['order'].m_orderRef)

        if not order_details and ib_order_id in self._tws.executions:
            executions = self._tws.executions[ib_order_id]
            last_exec_detail = list(executions.values())[-1]['exec_detail']
            last_exec_contract = list(executions.values())[-1]['contract']
            symbol = last_exec_contract.m_symbol
            order_details = self._parse_order_ref(last_exec_detail.m_orderRef)

        asset = self._safe_symbol_lookup(symbol)
        if not asset:
            log.warning(
                "Ignoring symbol {symbol} which has associated "
                "order but it is not registered in bundle".format(
                    symbol=symbol))
            return None

        if order_details:
            amount = self._action_qty_to_amount(order_details['action'],
                                                order_details['qty'])
            stop_price = order_details['stop_price']
            limit_price = order_details['limit_price']
            dt = order_details['dt']
        else:
            dt = pd.to_datetime('now', utc=True)
            amount, stop_price, limit_price = 0, None, None
            if ib_order_id in self._tws.open_orders:
                open_order = self._tws.open_orders[ib_order_id]['order']
                amount = self._action_qty_to_amount(
                    open_order.m_action, open_order.m_totalQuantity)
                stop_price = open_order.m_auxPrice
                limit_price = open_order.m_lmtPrice

        stop_price = None if stop_price == 0 else stop_price
        limit_price = None if limit_price == 0 else limit_price

        self._orders[zp_order_id] = ZPOrder(
            dt=dt,
            asset=asset,
            amount=amount,
            stop=stop_price,
            limit=limit_price,
            id=zp_order_id)
        self._orders[zp_order_id].broker_order_id = ib_order_id

        return self._orders[zp_order_id]

    @staticmethod
    def _ib_to_zp_status(ib_status):
        ib_status = ib_status.lower()
        if ib_status == 'submitted':
            return ZP_ORDER_STATUS.OPEN
        elif ib_status in ('pendingsubmit',
                           'pendingcancel',
                           'presubmitted'):
            return ZP_ORDER_STATUS.HELD
        elif ib_status == 'cancelled':
            return ZP_ORDER_STATUS.CANCELLED
        elif ib_status == 'filled':
            return ZP_ORDER_STATUS.FILLED
        elif ib_status == 'inactive':
            return ZP_ORDER_STATUS.REJECTED
        else:
            return None

    def _update_orders(self):
        def _update_from_order_status(zp_order, ib_order_id):
            if ib_order_id in self._tws.open_orders:
                open_order_state = self._tws.open_orders[ib_order_id]['state']

                zp_status = self._ib_to_zp_status(open_order_state.m_status)
                if zp_status is None:
                    log.warning(
                        "Order-{order_id}: "
                        "unknown order status: {order_status}.".format(
                            order_id=ib_order_id,
                            order_status=open_order_state.m_status))
                else:
                    zp_order.status = zp_status

            if ib_order_id in self._tws.order_statuses:
                order_status = self._tws.order_statuses[ib_order_id]

                zp_order.filled = order_status['filled']

                zp_status = self._ib_to_zp_status(order_status['status'])
                if zp_status:
                    zp_order.status = zp_status
                else:
                    log.warning("Order-{order_id}: "
                                "unknown order status: {order_status}."
                                .format(order_id=ib_order_id,
                                        order_status=order_status['status']))

        def _update_from_execution(zp_order, ib_order_id):
            if ib_order_id in self._tws.executions and \
                    ib_order_id not in self._tws.open_orders:
                zp_order.status = ZP_ORDER_STATUS.FILLED
                executions = self._tws.executions[ib_order_id]
                last_exec_detail = \
                    list(executions.values())[-1]['exec_detail']
                zp_order.filled = last_exec_detail.m_cumQty

        all_ib_order_ids = (set([e.broker_order_id
                                 for e in self._orders.values()]) |
                            set(self._tws.open_orders.keys()) |
                            set(self._tws.order_statuses.keys()) |
                            set(self._tws.executions.keys()) |
                            set(self._tws.commissions.keys()))
        for ib_order_id in all_ib_order_ids:
            zp_order = self._get_or_create_zp_order(ib_order_id)
            if zp_order:
                _update_from_execution(zp_order, ib_order_id)
                _update_from_order_status(zp_order, ib_order_id)

    @property
    def transactions(self):
        self._update_transactions()
        return self._transactions

    def _update_transactions(self):
        all_orders = list(self.orders.values())

        for ib_order_id, executions in iteritems(self._tws.executions):
            orders = [order
                      for order in all_orders
                      if order.broker_order_id == ib_order_id]

            if not orders:
                log.warning("No order found for executions: {}".format(
                    executions))
                continue

            assert len(orders) == 1
            order = orders[0]

            for exec_id, execution in iteritems(executions):
                if exec_id in self._transactions:
                    continue

                try:
                    commission = self._tws.commissions[ib_order_id][exec_id] \
                        .m_commission
                except KeyError:
                    log.warning(
                        "Commission not found for execution: {}".format(
                            exec_id))
                    commission = 0

                exec_detail = execution['exec_detail']
                is_buy = order.amount > 0
                amount = (exec_detail.m_shares if is_buy
                          else -1 * exec_detail.m_shares)
                tx = Transaction(
                    asset=order.asset,
                    amount=amount,
                    dt=pd.to_datetime(exec_detail.m_time, utc=True),
                    price=exec_detail.m_price,
                    order_id=order.id
                )
                self._transactions[exec_id] = tx

    def cancel_order(self, zp_order_id):
        ib_order_id = self.orders[zp_order_id].broker_order_id
        self._tws.cancelOrder(ib_order_id)

    def get_spot_value(self, assets, field, dt, data_frequency):
        symbol = str(assets.symbol)

        self.subscribe_to_market_data(assets)

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

    def get_last_traded_dt(self, asset):
        self.subscribe_to_market_data(asset)

        return self._tws.bars[asset.symbol].index[-1]

    def get_realtime_bars(self, assets, frequency):
        if frequency == '1m':
            resample_freq = '1 Min'
        elif frequency == '1d':
            resample_freq = '24 H'
        else:
            raise ValueError("Invalid frequency specified: %s" % frequency)

        df = pd.DataFrame()
        for asset in assets:
            symbol = str(asset.symbol)
            self.subscribe_to_market_data(asset)

            trade_prices = self._tws.bars[symbol]['last_trade_price']
            trade_sizes = self._tws.bars[symbol]['last_trade_size']
            ohlcv = trade_prices.resample(resample_freq).ohlc()
            ohlcv['volume'] = trade_sizes.resample(resample_freq).sum()

            # Add asset as level 0 column; ohlcv will be used as level 1 cols
            ohlcv.columns = pd.MultiIndex.from_product([[symbol, ],
                                                        ohlcv.columns])

            df = pd.concat([df, ohlcv], axis=1)

        return df
