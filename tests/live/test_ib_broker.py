import pandas as pd

# fix to allow zip_longest on Python 2.X and 3.X
try:                                    # Python 3
    from itertools import zip_longest
except ImportError:                     # Python 2
    from itertools import izip_longest as zip_longest
from math import fabs

from mock import patch, sentinel, Mock, MagicMock

from ib.ext.Contract import Contract
from ib.ext.Order import Order
from ib.ext.Execution import Execution
from ib.ext.OrderState import OrderState

from zipline.gens.brokers.ib_broker import IBBroker, TWSConnection
from zipline.testing.fixtures import WithSimParams
from zipline.finance.execution import (StopLimitOrder,
                                       MarketOrder,
                                       StopOrder,
                                       LimitOrder)
from zipline.finance.order import ORDER_STATUS
from zipline.testing.fixtures import (ZiplineTestCase,
                                      WithDataPortal)

class TestIBBroker(WithSimParams,
                   WithDataPortal,
                   ZiplineTestCase):
    ASSET_FINDER_EQUITY_SIDS = (1, 2)
    ASSET_FINDER_EQUITY_SYMBOLS = ("SPY", "XIV")

    @staticmethod
    def _tws_bars():
        with patch('zipline.gens.brokers.ib_broker.TWSConnection.connect'):
            tws = TWSConnection("localhost:9999:1111")

        tws._add_bar('SPY', 12.4, 10,
                     pd.to_datetime('2017-09-27 10:30:00', utc=True),
                     10, 12.401, False)
        tws._add_bar('SPY', 12.41, 10,
                     pd.to_datetime('2017-09-27 10:30:40', utc=True),
                     20, 12.411, False)
        tws._add_bar('SPY', 12.44, 20,
                     pd.to_datetime('2017-09-27 10:31:10', utc=True),
                     40, 12.441, False)
        tws._add_bar('SPY', 12.74, 5,
                     pd.to_datetime('2017-09-27 10:37:10', utc=True),
                     45, 12.741, True)
        tws._add_bar('SPY', 12.99, 15,
                     pd.to_datetime('2017-09-27 12:10:00', utc=True),
                     60, 12.991, False)
        tws._add_bar('XIV', 100.4, 100,
                     pd.to_datetime('2017-09-27 9:32:00', utc=True),
                     100, 100.401, False)
        tws._add_bar('XIV', 100.41, 100,
                     pd.to_datetime('2017-09-27 9:32:20', utc=True),
                     200, 100.411, True)
        tws._add_bar('XIV', 100.44, 200,
                     pd.to_datetime('2017-09-27 9:41:10', utc=True),
                     400, 100.441, False)
        tws._add_bar('XIV', 100.74, 50,
                     pd.to_datetime('2017-09-27 11:42:10', utc=True),
                     450, 100.741, False)

        return tws.bars

    @staticmethod
    def _create_contract(symbol):
        contract = Contract()
        contract.m_symbol = symbol
        contract.m_secType = 'STK'
        return contract

    @staticmethod
    def _create_order(action, qty, order_type, limit_price, stop_price):
        order = Order()
        order.m_action = action
        order.m_totalQuantity = qty
        order.m_auxPrice = stop_price
        order.m_lmtPrice = limit_price
        order.m_orderType = order_type
        return order

    @staticmethod
    def _create_order_state(status_):
        status = OrderState()
        status.m_status = status_
        return status

    @staticmethod
    def _create_exec_detail(order_id, shares, cum_qty, price, avg_price,
                            exec_time, exec_id):
        exec_detail = Execution()
        exec_detail.m_orderId = order_id
        exec_detail.m_shares = shares
        exec_detail.m_cumQty = cum_qty
        exec_detail.m_price = price
        exec_detail.m_avgPrice = avg_price
        exec_detail.m_time = exec_time
        exec_detail.m_execId = exec_id
        return exec_detail

    @patch('zipline.gens.brokers.ib_broker.TWSConnection')
    def test_get_spot_value(self, tws):
        dt = None  # dt is not used in real broker
        data_freq = 'minute'
        asset = self.asset_finder.retrieve_asset(1)
        bars = {'last_trade_price': [12, 10, 11, 14],
                'last_trade_size': [1, 2, 3, 4],
                'total_volume': [10, 10, 10, 10],
                'vwap': [12.1, 10.1, 11.1, 14.1],
                'single_trade_flag': [0, 1, 0, 1]}
        last_trade_times = [pd.to_datetime('2017-06-16 10:30:00', utc=True),
                            pd.to_datetime('2017-06-16 10:30:11', utc=True),
                            pd.to_datetime('2017-06-16 10:30:30', utc=True),
                            pd.to_datetime('2017-06-17 10:31:9', utc=True)]
        index = pd.DatetimeIndex(last_trade_times)
        broker = IBBroker(sentinel.tws_uri)
        tws.return_value.bars = {asset.symbol: pd.DataFrame(
            index=index, data=bars)}

        price = broker.get_spot_value(asset, 'price', dt, data_freq)
        last_trade = broker.get_spot_value(asset, 'last_traded', dt, data_freq)
        open_ = broker.get_spot_value(asset, 'open', dt, data_freq)
        high = broker.get_spot_value(asset, 'high', dt, data_freq)
        low = broker.get_spot_value(asset, 'low', dt, data_freq)
        close = broker.get_spot_value(asset, 'close', dt, data_freq)
        volume = broker.get_spot_value(asset, 'volume', dt, data_freq)

        # Only the last minute is taken into account, therefore
        # the first bar is ignored
        assert price == bars['last_trade_price'][-1]
        assert last_trade == last_trade_times[-1]
        assert open_ == bars['last_trade_price'][1]
        assert high == max(bars['last_trade_price'][1:])
        assert low == min(bars['last_trade_price'][1:])
        assert close == bars['last_trade_price'][-1]
        assert volume == sum(bars['last_trade_size'][1:])

    def test_get_realtime_bars_produces_correct_df(self):
        bars = self._tws_bars()

        with patch('zipline.gens.brokers.ib_broker.TWSConnection'):
            broker = IBBroker(sentinel.tws_uri)
            broker._tws.bars = bars

        assets = (self.asset_finder.retrieve_asset(1),
                  self.asset_finder.retrieve_asset(2))

        realtime_history = broker.get_realtime_bars(assets, '1m')

        asset_spy = self.asset_finder.retrieve_asset(1)
        asset_xiv = self.asset_finder.retrieve_asset(2)

        assert asset_spy in realtime_history
        assert asset_xiv in realtime_history

        spy = realtime_history[asset_spy]
        xiv = realtime_history[asset_xiv]

        assert list(spy.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert list(xiv.columns) == ['open', 'high', 'low', 'close', 'volume']

        # There are 159 minutes between the first (XIV @ 2017-09-27 9:32:00)
        # and the last bar (SPY @ 2017-09-27 12:10:00)
        assert len(realtime_history) == 159

        spy_non_na = spy.dropna()
        xiv_non_na = xiv.dropna()
        assert len(spy_non_na) == 4
        assert len(xiv_non_na) == 3

        assert spy_non_na.iloc[0].name == pd.to_datetime(
            '2017-09-27 10:30:00', utc=True)
        assert spy_non_na.iloc[0].open == 12.40
        assert spy_non_na.iloc[0].high == 12.41
        assert spy_non_na.iloc[0].low == 12.40
        assert spy_non_na.iloc[0].close == 12.41
        assert spy_non_na.iloc[0].volume == 20

        assert spy_non_na.iloc[1].name == pd.to_datetime(
            '2017-09-27 10:31:00', utc=True)
        assert spy_non_na.iloc[1].open == 12.44
        assert spy_non_na.iloc[1].high == 12.44
        assert spy_non_na.iloc[1].low == 12.44
        assert spy_non_na.iloc[1].close == 12.44
        assert spy_non_na.iloc[1].volume == 20

        assert spy_non_na.iloc[-1].name == pd.to_datetime(
            '2017-09-27 12:10:00', utc=True)
        assert spy_non_na.iloc[-1].open == 12.99
        assert spy_non_na.iloc[-1].high == 12.99
        assert spy_non_na.iloc[-1].low == 12.99
        assert spy_non_na.iloc[-1].close == 12.99
        assert spy_non_na.iloc[-1].volume == 15

        assert xiv_non_na.iloc[0].name == pd.to_datetime(
            '2017-09-27 9:32:00', utc=True)
        assert xiv_non_na.iloc[0].open == 100.4
        assert xiv_non_na.iloc[0].high == 100.41
        assert xiv_non_na.iloc[0].low == 100.4
        assert xiv_non_na.iloc[0].close == 100.41
        assert xiv_non_na.iloc[0].volume == 200

    @patch('zipline.gens.brokers.ib_broker.symbol_lookup')
    def test_new_order_appears_in_orders(self, symbol_lookup):
        with patch('zipline.gens.brokers.ib_broker.TWSConnection.connect'):
            broker = IBBroker("localhost:9999:1111", account_id='TEST-123')
            broker._tws.nextValidId(0)

        asset = self.asset_finder.retrieve_asset(1)
        symbol_lookup.return_value = asset
        amount = -4
        limit_price = 43.1
        stop_price = 6
        style = StopLimitOrder(limit_price=limit_price, stop_price=stop_price)
        order = broker.order(asset, amount, style)

        assert len(broker.orders) == 1
        assert broker.orders[order.id] == order
        assert order.open
        assert order.asset == asset
        assert order.amount == amount
        assert order.limit == limit_price
        assert order.stop == stop_price
        assert (order.dt - pd.to_datetime('now', utc=True) <
                pd.Timedelta('10s'))

    @patch('zipline.gens.brokers.ib_broker.symbol_lookup')
    def test_orders_loaded_from_open_orders(self, symbol_lookup):
        with patch('zipline.gens.brokers.ib_broker.TWSConnection.connect'):
            broker = IBBroker("localhost:9999:1111", account_id='TEST-123')

        asset = self.asset_finder.retrieve_asset(1)
        symbol_lookup.return_value = asset

        ib_order_id = 3
        ib_contract = self._create_contract(str(asset.symbol))
        action, qty, order_type, limit_price, stop_price = \
            'SELL', 40, 'STP LMT', 4.3, 2
        ib_order = self._create_order(
            action, qty, order_type, limit_price, stop_price)
        ib_state = self._create_order_state('PreSubmitted')
        broker._tws.openOrder(ib_order_id, ib_contract, ib_order, ib_state)

        assert len(broker.orders) == 1
        zp_order = list(broker.orders.values())[-1]
        assert zp_order.broker_order_id == ib_order_id
        assert zp_order.status == ORDER_STATUS.HELD
        assert zp_order.open
        assert zp_order.asset == asset
        assert zp_order.amount == -40
        assert zp_order.limit == limit_price
        assert zp_order.stop == stop_price
        assert (zp_order.dt - pd.to_datetime('now', utc=True) <
                pd.Timedelta('10s'))

        @patch('zipline.gens.brokers.ib_broker.symbol_lookup')
        def test_orders_loaded_from_exec_details(self, symbol_lookup):
            with patch('zipline.gens.brokers.ib_broker.TWSConnection.connect'):
                broker = IBBroker("localhost:9999:1111", account_id='TEST-123')

            asset = self.asset_finder.retrieve_asset(1)
            symbol_lookup.return_value = asset

            (req_id, ib_order_id, shares, cum_qty,
             price, avg_price, exec_time, exec_id) = (7, 3, 12, 40,
                                                      12.43, 12.50,
                                                      '20160101 14:20', 4)
            ib_contract = self._create_contract(str(asset.symbol))
            exec_detail = self._create_exec_detail(
                ib_order_id, shares, cum_qty, price, avg_price,
                exec_time, exec_id)
            broker._tws.execDetails(req_id, ib_contract, exec_detail)

            assert len(broker.orders) == 1
            zp_order = list(broker.orders.values())[-1]
            assert zp_order.broker_order_id == ib_order_id
            assert zp_order.open
            assert zp_order.asset == asset
            assert zp_order.amount == -40
            assert zp_order.limit == limit_price
            assert zp_order.stop == stop_price
            assert (zp_order.dt - pd.to_datetime('now', utc=True) <
                    pd.Timedelta('10s'))

    @patch('zipline.gens.brokers.ib_broker.symbol_lookup')
    def test_orders_updated_from_order_status(self, symbol_lookup):
        with patch('zipline.gens.brokers.ib_broker.TWSConnection.connect'):
            broker = IBBroker("localhost:9999:1111", account_id='TEST-123')
            broker._tws.nextValidId(0)

        # orderStatus calls only work if a respective order has been created
        asset = self.asset_finder.retrieve_asset(1)
        symbol_lookup.return_value = asset
        amount = -4
        limit_price = 43.1
        stop_price = 6
        style = StopLimitOrder(limit_price=limit_price, stop_price=stop_price)
        order = broker.order(asset, amount, style)

        ib_order_id = order.broker_order_id
        status = 'Filled'
        filled = 14
        remaining = 9
        avg_fill_price = 12.4
        perm_id = 99
        parent_id = 88
        last_fill_price = 12.3
        client_id = 1111
        why_held = ''

        broker._tws.orderStatus(ib_order_id,
                                status, filled, remaining, avg_fill_price,
                                perm_id, parent_id, last_fill_price, client_id,
                                why_held)

        assert len(broker.orders) == 1
        zp_order = list(broker.orders.values())[-1]
        assert zp_order.broker_order_id == ib_order_id
        assert zp_order.status == ORDER_STATUS.FILLED
        assert not zp_order.open
        assert zp_order.asset == asset
        assert zp_order.amount == amount
        assert zp_order.limit == limit_price
        assert zp_order.stop == stop_price
        assert (zp_order.dt - pd.to_datetime('now', utc=True) <
                pd.Timedelta('10s'))

    @patch('zipline.gens.brokers.ib_broker.symbol_lookup')
    def test_multiple_orders(self, symbol_lookup):
        with patch('zipline.gens.brokers.ib_broker.TWSConnection.connect'):
            broker = IBBroker("localhost:9999:1111", account_id='TEST-123')
            broker._tws.nextValidId(0)

        asset = self.asset_finder.retrieve_asset(1)
        symbol_lookup.return_value = asset

        order_count = 0
        for amount, order_style in [
                (-112, StopLimitOrder(limit_price=9, stop_price=1)),
                (43, LimitOrder(limit_price=10)),
                (-99, StopOrder(stop_price=8)),
                (-32, MarketOrder())]:
            order = broker.order(asset, amount, order_style)
            order_count += 1

            assert order_count == len(broker.orders)
            assert broker.orders[order.id] == order
            is_buy = amount > 0
            assert order.stop == order_style.get_stop_price(is_buy)
            assert order.limit == order_style.get_limit_price(is_buy)

    def test_order_ref_serdes(self):
        # Even though _creater_order_ref and _parse_order_ref is private
        # it is helpful to test as it plays a key role to re-create orders
        order = self._create_order("BUY", 66, "STP LMT", 13.4, 44.2)
        serialized = IBBroker._create_order_ref(order)
        deserialized = IBBroker._parse_order_ref(serialized)
        assert deserialized['action'] == order.m_action
        assert deserialized['qty'] == order.m_totalQuantity
        assert deserialized['order_type'] == order.m_orderType
        assert deserialized['limit_price'] == order.m_lmtPrice
        assert deserialized['stop_price'] == order.m_auxPrice
        assert (deserialized['dt'] - pd.to_datetime('now', utc=True) <
                pd.Timedelta('10s'))

    @patch('zipline.gens.brokers.ib_broker.symbol_lookup')
    def test_transactions_not_created_for_incompl_orders(self, symbol_lookup):
        with patch('zipline.gens.brokers.ib_broker.TWSConnection.connect'):
            broker = IBBroker("localhost:9999:1111", account_id='TEST-123')
            broker._tws.nextValidId(0)

        asset = self.asset_finder.retrieve_asset(1)
        symbol_lookup.return_value = asset
        amount = -4
        limit_price = 43.1
        stop_price = 6
        style = StopLimitOrder(limit_price=limit_price, stop_price=stop_price)
        order = broker.order(asset, amount, style)
        assert not broker.transactions
        assert len(broker.orders) == 1
        assert broker.orders[order.id].open

        ib_order_id = order.broker_order_id
        ib_contract = self._create_contract(str(asset.symbol))
        action, qty, order_type, limit_price, stop_price = \
            'SELL', 4, 'STP LMT', 4.3, 2
        ib_order = self._create_order(
            action, qty, order_type, limit_price, stop_price)
        ib_state = self._create_order_state('PreSubmitted')
        broker._tws.openOrder(ib_order_id, ib_contract, ib_order, ib_state)

        broker._tws.orderStatus(ib_order_id, status='Cancelled', filled=0,
                                remaining=4, avg_fill_price=0.0, perm_id=4,
                                parent_id=4, last_fill_price=0.0, client_id=32,
                                why_held='')
        assert not broker.transactions
        assert len(broker.orders) == 1
        assert not broker.orders[order.id].open

        broker._tws.orderStatus(ib_order_id, status='Inactive', filled=0,
                                remaining=4, avg_fill_price=0.0, perm_id=4,
                                parent_id=4, last_fill_price=0.0,
                                client_id=1111, why_held='')
        assert not broker.transactions
        assert len(broker.orders) == 1
        assert not broker.orders[order.id].open

    @patch('zipline.gens.brokers.ib_broker.symbol_lookup')
    def test_transactions_created_for_complete_orders(self, symbol_lookup):
        with patch('zipline.gens.brokers.ib_broker.TWSConnection.connect'):
            broker = IBBroker("localhost:9999:1111", account_id='TEST-123')
            broker._tws.nextValidId(0)

        asset = self.asset_finder.retrieve_asset(1)
        symbol_lookup.return_value = asset

        order_count = 0
        for amount, order_style in [
                (-112, StopLimitOrder(limit_price=9, stop_price=1)),
                (43, LimitOrder(limit_price=10)),
                (-99, StopOrder(stop_price=8)),
                (-32, MarketOrder())]:
            order = broker.order(asset, amount, order_style)
            broker._tws.orderStatus(order.broker_order_id, 'Filled',
                                    filled=int(fabs(amount)), remaining=0,
                                    avg_fill_price=111, perm_id=0, parent_id=1,
                                    last_fill_price=112, client_id=1111,
                                    why_held='')
            contract = self._create_contract(str(asset.symbol))
            (shares, cum_qty, price, avg_price, exec_time, exec_id) = \
                (int(fabs(amount)), int(fabs(amount)), 12.3, 12.31,
                 pd.to_datetime('now', utc=True), order_count)
            exec_detail = self._create_exec_detail(
                order.broker_order_id, shares, cum_qty,
                price, avg_price, exec_time, exec_id)
            broker._tws.execDetails(0, contract, exec_detail)
            order_count += 1

            assert len(broker.transactions) == order_count
            transactions = [tx
                            for tx in broker.transactions.values()
                            if tx.order_id == order.id]
            assert len(transactions) == 1

            assert broker.transactions[exec_id].asset == asset
            assert broker.transactions[exec_id].amount == order.amount
            assert (broker.transactions[exec_id].dt -
                    pd.to_datetime('now', utc=True) < pd.Timedelta('10s'))
            assert broker.transactions[exec_id].price == price
            assert broker.orders[order.id].commission == 0
