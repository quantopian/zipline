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

    def __init__(self):
        self._api = tradeapi.REST()

    def subscribe_to_market_data(self, asset):
        '''Do nothing to comply the interface'''
        pass

    def subscribed_assets(self):
        '''Do nothing to comply the interface'''
        return []
      
    def set_metrics_tracker(self, metrics_tracker):
        self.metrics_tracker = metrics_tracker

    @property
    def positions(self):
        self._get_positions_from_broker()
        return self.metrics_tracker.positions


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
        orders = {}
        for o in self._api.list_orders('all'):
            try:
                orders[o.client_order_id] = self._order2zp(o)
            except:
                continue
        return orders

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
            try:
                last_trade = self._api.get_last_trade(symbols[0])
                return last_trade.price
            except:
                return np.nan

        bars = self._api.get_barset(symbols, '1Min', limit=1).df
        if bars.empty:
            return np.nan
        if not np.isnan(bars[assets.symbol][field]).all():
            return float(bars[assets.symbol][field])
        # if assets_is_scalar:
        #     if len(bars_list) == 0:
        #         return np.nan
        #     return bars_list[0].bars[-1]._raw[field]
        # bars_map = {a.symbol: a for a in bars_list}
        # return [
        #     bars_map[symbol].bars[-1]._raw[field]
        #     for symbol in symbols
        # ]

    def _get_positions_from_broker(self):
        """
        get the positions from the broker and update zipline objects ( the ledger )
        should be used once at startup and once every time we want to refresh the positions array
        """
        cur_pos_in_tracker = self.metrics_tracker.positions
        positions = self._api.list_positions()
        for ap_position in positions:
            # ap_position = positions[symbol]
            try:
                z_position = zp.Position(zp.InnerPosition(symbol_lookup(ap_position.symbol)))
                editable_position = zp.MutableView(z_position)
            except SymbolNotFound:
                # The symbol might not have been ingested to the db therefore
                # it needs to be skipped.
                log.warning('Wanted to subscribe to %s, but this asset is probably not ingested' % ap_position.symbol)
                continue
            if int(ap_position.qty) == 0:
                continue
            editable_position._underlying_position.amount = int(ap_position.qty)
            editable_position._underlying_position.cost_basis = float(ap_position.avg_entry_price)
            editable_position._underlying_position.last_sale_price = float(ap_position.current_price)
            editable_position._underlying_position.last_sale_date = self._api.get_last_trade(ap_position.symbol).timestamp
            
            self.metrics_tracker.update_position(z_position.asset,
                                                 amount=z_position.amount,
                                                 last_sale_price=z_position.last_sale_price,
                                                 last_sale_date=z_position.last_sale_date,
                                                 cost_basis=z_position.cost_basis)

        # now let's sync the positions in the internal zipline objects
        position_names = [p.symbol for p in positions]
        assets_to_update = []  # separate list to not change list while iterating
        for asset in cur_pos_in_tracker:
            if asset.symbol not in position_names:
                assets_to_update.append(asset)
        for asset in assets_to_update:
            # deleting object from the metrics_tracker as its not in the portfolio
            self.metrics_tracker.update_position(asset,
                                                 amount=0)
        # for some reason, the metrics tracker has self.positions AND self.portfolio.positions. let's make sure
        # these objects are consistent
        self.metrics_tracker._ledger._portfolio.positions = self.metrics_tracker.positions                                                 

    def get_realtime_bars(self, assets, data_frequency):
        # TODO: cache the result. The caller
        # (DataPortalLive#get_history_window) makes use of only one
        # column at a time.
        assets_is_scalar = not isinstance(assets, (list, set, tuple, pd.Index))
        is_daily = 'd' in data_frequency  # 'daily' or '1d'
        if assets_is_scalar:
            symbols = [assets.symbol]
        else:
            symbols = [asset.symbol for asset in assets]
        timeframe = '1D' if is_daily else '1Min'
        df = self._api.get_barset(symbols, timeframe, limit=500).df
        if not is_daily:
            df = df.between_time("09:30", "16:00")
        return df
