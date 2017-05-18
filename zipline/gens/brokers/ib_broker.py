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

from collections import namedtuple, defaultdict
from time import sleep

import pandas as pd

from zipline.gens.brokers.broker import Broker

import zipline.protocol as zp
from zipline.api import symbol as symbol_lookup
from zipline.errors import SymbolNotFound

from ib.ext.EClientSocket import EClientSocket
from ib.ext.EWrapper import EWrapper
from ib.ext.Contract import Contract

from logbook import Logger

log = Logger('IB Broker')


RTVolumeBar = namedtuple('RTVolumeBar', ['last_trade_price',
                                         'last_trade_size',
                                         'last_trade_time',
                                         'total_volume',
                                         'vwap',
                                         'single_trade_flag'])

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
    def __init__(self, tws_uri):
        EWrapper.__init__(self)
        EClientSocket.__init__(self, anyWrapper=self)

        self.tws_uri = tws_uri
        host, port, client_id = self.tws_uri.split(':')

        self._next_ticker_id = 0
        self.managed_accounts = None
        self.ticker_id_to_symbol = {}
        self.last_tick = defaultdict(dict)
        self.realtime_bars = {}
        self.accounts = {}
        self.accounts_download_complete = False
        self.positions = {}
        self.portfolio = {}
        self.time_skew = None

        log.info("Connecting: {}:{}:{}".format(host, int(port),
                                               int(client_id)))
        self.eConnect(host, int(port), int(client_id))

        self._download_account_details()
        log.info("Managed accounts: {}".format(self.managed_accounts))

        self.reqCurrentTime()
        while self.time_skew is None:
            sleep(0.1)
        log.info("Local - Server Time Skew: {}".format(self.time_skew))

    def _download_account_details(self):
        self.reqManagedAccts()
        while not self.managed_accounts:
            sleep(0.1)

        for account in self.managed_accounts:
            self.reqAccountUpdates(subscribe=True, acctCode=account)
        while not self.accounts_download_complete:
            sleep(0.1)

    def _get_next_ticker_id(self):
        ticker_id = self._next_ticker_id
        self._next_ticker_id += 1
        return ticker_id

    def subscribe_to_market_data(self,
                                 symbol,
                                 sec_type='STK',
                                 exchange='SMART',
                                 currency='USD'):
        contract = Contract()
        contract.m_symbol = symbol
        contract.m_secType = sec_type
        contract.m_exchange = exchange
        contract.m_currency = currency

        ticker_id = self._get_next_ticker_id

        self.ticker_id_to_symbol[ticker_id] = symbol

        tick_list = "233"  # RTVolume, return tick_type == 48
        self.reqMktData(self._get_next_ticker_id, contract, tick_list, False)

    def _process_tick(self, ticker_id, tick_type, value):
        try:
            instr = self.ticker_id_to_symbol[ticker_id]
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

            rt_volume_bar = RTVolumeBar(last_trade_price=float(
                                        last_trade_price),
                                        last_trade_size=int(last_trade_size),
                                        last_trade_time=float(last_trade_time),
                                        total_volume=int(total_volume),
                                        vwap=float(vwap),
                                        single_trade_flag=single_trade_flag)
            log.debug("RT Volume Bar: {}".format(rt_volume_bar))
            self.realtime_bars[instr].append(rt_volume_bar)

    def tickPrice(self, ticker_id, field, price, can_auto_execute):
        log_message('tickPrice', vars())
        self._process_tick(ticker_id, tick_type=field, value=price)

    def tickSize(self, ticker_id, field, size):
        log_message('tickSize', vars())
        self._process_tick(ticker_id, tick_type=field, value=size)

    def tickOptionComputation(self,
                              ticker_id, field, implied_vol, delta, opt_price,
                              pv_dividend, gamma, vega, theta, und_price):
        log_message('tickOptionComputation', vars())

    def tickGeneric(self, ticker_id, tick_type, value):
        log_message('tickGeneric', vars())
        self._process_tick(ticker_id, tick_type=tick_type, value=value)

    def tickString(self, ticker_id, tick_type, value):
        log_message('tickString', vars())
        self._process_tick(ticker_id, tick_type=tick_type, value=value)

    def tickEFP(self, ticker_id, tick_type, basis_points,
                formatted_basis_points, implied_future, hold_days,
                future_expiry, dividend_impact, dividends_to_expiry):
        log_message('tickEFP', vars())

    def orderStatus(self, order_id, status, filled, remaining, avg_fill_price,
                    perm_id, parent_id, last_fill_price, client_id, why_held):
        log_message('orderStatus', vars())

    def openOrder(self, order_id, contract, order, state):
        log_message('openOrder', vars())

    def openOrderEnd(self):
        log_message('openOrderEnd', vars())

    def updateAccountValue(self, key, value, currency, account_name):
        log_message('updateAccountValue', vars())
        self.accounts.setdefault(account_name, {})
        self.accounts[account_name].setdefault(currency, {})
        self.accounts[account_name][currency].setdefault(key, {})

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
        log_message('updatePortfolio', vars())

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
        log_message('updateAccountTime', vars())

    def accountDownloadEnd(self, account_name):
        log_message('accountDownloadEnd', vars())
        self.accounts_download_complete = True

    def nextValidId(self, order_id):
        log_message('nextValidId', vars())

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
        log_message('error', vars())

    def updateMktDepth(self, ticker_id, position, operation, side, price,
                       size):
        log_message('updateMktDepth', vars())

    def updateMktDepthL2(self, ticker_id, position, market_maker, operation,
                         side, price, size):
        log_message('updateMktDepthL2', vars())

    def updateNewsBulletin(self, msg_id, msg_type, message, orig_exchange):
        log_message('updateNewsBulletin', vars())

    def managedAccounts(self, accounts_list):
        log_message('managedAccounts', vars())
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
        log_message('currentTime', vars())
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
        self._tws = TWSConnection(tws_uri)
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
                log.warn("Symbol {} not found in the data base.", symbol)
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
        raise NotImplementedError()

    def get_open_orders(self, asset):
        raise NotImplementedError()

    def get_order(self, order_id):
        raise NotImplementedError()

    def cancel_order(self, order_param):
        raise NotImplementedError()

    def get_spot_value(self, assets, field, dt, data_frequency):
        raise NotImplementedError()
