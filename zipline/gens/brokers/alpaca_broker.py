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

import alpaca_trade_api as tradeapi
from zipline.gens.brokers.broker import Broker
import zipline.protocol as zp
from zipline.finance.order import (Order as ZPOrder,
                                   ORDER_STATUS as ZP_ORDER_STATUS)
from zipline.finance.execution import (MarketOrder,
                                       LimitOrder,
                                       StopOrder,
                                       StopLimitOrder)
from zipline.finance.transaction import Transaction
from zipline.api import symbol as symbol_lookup
from zipline.errors import SymbolNotFound
import pandas as pd
import numpy as np
import uuid

from logbook import Logger
import sys

if sys.version_info > (3,):
    long = int

log = Logger('Alpaca Broker')
NY = 'America/New_York'


class ALPACABroker(Broker):
    '''
    Broker class for Alpaca.
    The uri parameter is not used. Instead, the API key must be
    set via environment variables (APCA_API_KEY_ID and APCA_API_SECRET_KEY).
    Orders are identified by the UUID (v4) generated here and
    associated in the broker side using client_order_id attribute.
    Currently this class makes use of REST API only, but websocket
    streaming can possibly used too.
    '''

    def __init__(self, uri):
        self._api = tradeapi.REST()

    def subscribe_to_market_data(self, asset):
        '''Do nothing to comply the interface'''
        pass

    def subscribed_assets(self):
        '''Do nothing to comply the interface'''
        return []

    @property
    def positions(self):
        z_positions = zp.Positions()
        positions = self._api.list_positions()
        position_map = {}
        symbols = []
        for pos in positions:
            symbol = pos.symbol
            try:
                z_position = zp.Position(symbol_lookup(symbol))
            except SymbolNotFound:
                continue
            z_position.amount = pos.qty
            z_position.cost_basis = float(pos.cost_basis)
            z_position.last_sale_price = None
            z_position.last_sale_date = None
            z_positions[symbol_lookup(symbol)] = z_position
            symbols.append(symbol)
            position_map[symbol] = z_position

        quotes = self._api.list_quotes(symbols)
        for quote in quotes:
            price = quote.last
            dt = quote.last_timestamp
            z_position = position_map[quote.symbol]
            z_position.last_sale_price = float(price)
            z_position.last_sale_date = dt
        return z_positions

    @property
    def portfolio(self):
        account = self._api.get_account()
        z_portfolio = zp.Portfolio()
        z_portfolio.cash = float(account.cash)
        z_portfolio.positions = self.positions
        z_portfolio.positions_value = float(
            account.portfolio_value) - float(account.cash)
        z_portfolio.portfolio_value = float(account.portfolio_value)
        return z_portfolio

    @property
    def account(self):
        account = self._api.get_account()
        z_account = zp.Account()
        z_account.buying_power = float(account.cash)
        z_account.total_position_value = float(
            account.portfolio_value) - float(account.cash)
        z_account.net_liquidation = account.portfolio_value
        return z_account

    @property
    def time_skew(self):
        return pd.Timedelta('0 sec')  # TODO: use clock API

    def is_alive(self):
        try:
            self._api.get_account()
            return True
        except BaseException:
            return False

    def _order2zp(self, order):
        zp_order = ZPOrder(
            id=order.client_order_id,
            asset=symbol_lookup(order.symbol),
            amount=int(order.qty) if order.side == 'buy' else -int(order.qty),
            stop=float(order.stop_price) if order.stop_price else None,
            limit=float(order.limit_price) if order.limit_price else None,
            dt=order.submitted_at,
            commission=0,
        )
        zp_order.status = ZP_ORDER_STATUS.OPEN
        if order.canceled_at:
            zp_order.status = ZP_ORDER_STATUS.CANCELLED
        if order.failed_at:
            zp_order.status = ZP_ORDER_STATUS.REJECTED
        if order.filled_at:
            zp_order.status = ZP_ORDER_STATUS.FILLED
            zp_order.filled = int(order.filled_qty)
        return zp_order

    def _new_order_id(self):
        return uuid.uuid4().hex

    def order(self, asset, amount, style):
        symbol = asset.symbol
        qty = amount if amount > 0 else -amount
        side = 'buy' if amount > 0 else 'sell'
        order_type = 'market'
        if isinstance(style, MarketOrder):
            order_type = 'market'
        elif isinstance(style, LimitOrder):
            order_type = 'limit'
        elif isinstance(style, StopOrder):
            order_type = 'stop'
        elif isinstance(style, StopLimitOrder):
            order_type = 'stop_limit'

        limit_price = style.get_limit_price(side == 'buy') or None
        stop_price = style.get_stop_price(side == 'buy') or None

        zp_order_id = self._new_order_id()
        dt = pd.to_datetime('now', utc=True)
        zp_order = ZPOrder(
            dt=dt,
            asset=asset,
            amount=amount,
            stop=stop_price,
            limit=limit_price,
            id=zp_order_id,
        )

        order = self._api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force='day',
            limit_price=limit_price,
            stop_price=stop_price,
            client_order_id=zp_order.id,
        )
        zp_order = self._order2zp(order)
        return zp_order

    @property
    def orders(self):
        return {
            o.client_order_id: self._order2zp(o)
            for o in self._api.list_orders('all')
        }

    @property
    def transactions(self):
        orders = self._api.list_orders(status='closed')
        results = {}
        for order in orders:
            if order.filled_at is None:
                continue
            tx = Transaction(
                asset=symbol_lookup(order.symbol),
                amount=int(order.filled_qty),
                dt=order.filled_at,
                price=float(order.filled_avg_price),
                order_id=order.client_order_id)
            results[order.client_order_id] = tx
        return results

    def cancel_order(self, zp_order_id):
        try:
            order = self._api.get_order_by_client_order_id(zp_order_id)
            self._api.cancel_order(order.id)
        except Exception as e:
            log.error(e)
            return

    def get_last_traded_dt(self, asset):
        quote = self._api.get_quote(asset.symbol)
        return pd.Timestamp(quote.last_timestamp)

    def get_spot_value(self, assets, field, dt, data_frequency):
        assert(field in (
            'open', 'high', 'low', 'close', 'volume', 'price', 'last_traded'))
        assets_is_scalar = not isinstance(assets, (list, set, tuple))
        if assets_is_scalar:
            symbols = [assets.symbol]
        else:
            symbols = [asset.symbol for asset in assets]
        if field in ('price', 'last_traded'):
            quotes = self._api.list_quotes(symbols)
            if assets_is_scalar:
                if field == 'price':
                    if len(quotes) == 0:
                        return np.nan
                    return quotes[-1].last
                else:
                    if len(quotes) == 0:
                        return pd.NaT
                    return quotes[-1].last_timestamp
            else:
                return [
                    quote.last if field == 'price' else quote.last_timestamp
                    for quote in quotes
                ]

        bars_list = self._api.list_bars(symbols, '1Min', limit=1)
        if assets_is_scalar:
            if len(bars_list) == 0:
                return np.nan
            return bars_list[0].bars[-1]._raw[field]
        bars_map = {a.symbol: a for a in bars_list}
        return [
            bars_map[symbol].bars[-1]._raw[field]
            for symbol in symbols
        ]

    def get_realtime_bars(self, assets, data_frequency):
        # TODO: cache the result. The caller
        # (DataPortalLive#get_history_window) makes use of only one
        # column at a time.
        assets_is_scalar = not isinstance(assets, (list, set, tuple))
        is_daily = 'd' in data_frequency  # 'daily' or '1d'
        if assets_is_scalar:
            symbols = [assets.symbol]
        else:
            symbols = [asset.symbol for asset in assets]
        timeframe = '1D' if is_daily else '1Min'

        bars_list = self._api.list_bars(symbols, timeframe, limit=500)
        bars_map = {a.symbol: a for a in bars_list}
        dfs = []
        for asset in assets if not assets_is_scalar else [assets]:
            symbol = asset.symbol
            df = bars_map[symbol].df.copy()
            if df.index.tz is None:
                df.index = df.index.tz_localize(
                    'utc').tz_convert('America/New_York')
            df.columns = pd.MultiIndex.from_product([[asset, ], df.columns])
            dfs.append(df)
        return pd.concat(dfs, axis=1)
