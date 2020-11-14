import unittest
import pandas as pd

# fix to allow zip_longest on Python 2.X and 3.X
try:                                    # Python 3
    from itertools import zip_longest
except ImportError:                     # Python 2
    from itertools import izip_longest as zip_longest

from mock import patch
import alpaca_trade_api.rest as apca

from zipline.gens.brokers.alpaca_broker import ALPACABroker
from zipline.testing.fixtures import WithSimParams
from zipline.finance.execution import (StopLimitOrder,
                                       MarketOrder,
                                       StopOrder,
                                       LimitOrder)
from zipline.finance.order import ORDER_STATUS
from zipline.testing.fixtures import (ZiplineTestCase,
                                      WithDataPortal)


@unittest.skip("Failing on CI - fix later")
class TestALPACABroker(WithSimParams,
                       WithDataPortal,
                       ZiplineTestCase):
    ASSET_FINDER_EQUITY_SIDS = (1, 2)
    ASSET_FINDER_EQUITY_SYMBOLS = ("SPY", "XIV")

    @patch('zipline.gens.brokers.alpaca_broker.tradeapi')
    def test_get_realtime_bars(self, tradeapi):
        api = tradeapi.REST()
        bars = [
            {
                'time': '2017-06-17T10:31:00-0400',
                'open': 102.0,
                'high': 102.5,
                'low': 101.5,
                'close': 102.1,
                'volume': 998,
            },
            {
                'time': '2017-06-17T10:32:00-0400',
                'open': 103.0,
                'high': 103.5,
                'low': 102.5,
                'close': 103.1,
                'volume': 996,
            },
        ]
        api.list_bars.return_value = [
            apca.AssetBars({
                'symbol': 'SPY',
                'bars': bars,
            })
        ]
        broker = ALPACABroker('')
        asset = self.asset_finder.retrieve_asset(1)
        ret = broker.get_realtime_bars(asset, '1m')
        assert ret[asset, 'open'].values[0] == 102.0

        ret = broker.get_realtime_bars([asset], '1m')
        assert ret[asset, 'close'].values[1] == 103.1

    @patch('zipline.gens.brokers.alpaca_broker.tradeapi')
    def test_get_spot_value(self, tradeapi):
        api = tradeapi.REST()

        dt = None  # dt is not used in real broker
        data_freq = 'minute'
        asset = self.asset_finder.retrieve_asset(1)
        bar = {'time': '2017-06-17T10:31:09-0400',
               'open': 103.0,
               'high': 103.5,
               'low': 102.5,
               'close': 103.1,
               'volume': 996}
        broker = ALPACABroker('')
        api.list_bars.return_value = [
            apca.AssetBars({
                'symbol': 'SPY',
                'bars': [bar],
            })
        ]

        quote = {
            'last': 103.8,
            'last_timestamp': '2017-06-17T10:31:13-0400',
        }
        api.list_quotes.return_value = [
            apca.Quote(quote)
        ]

        price = broker.get_spot_value(asset, 'price', dt, data_freq)
        last_trade = broker.get_spot_value(asset, 'last_traded', dt, data_freq)
        open_ = broker.get_spot_value(asset, 'open', dt, data_freq)
        high = broker.get_spot_value(asset, 'high', dt, data_freq)
        low = broker.get_spot_value(asset, 'low', dt, data_freq)
        close = broker.get_spot_value(asset, 'close', dt, data_freq)
        volume = broker.get_spot_value(asset, 'volume', dt, data_freq)

        assert price == quote['last']
        assert last_trade == pd.Timestamp(quote['last_timestamp'])
        assert open_ == bar['open']
        assert high == bar['high']
        assert low == bar['low']
        assert close == bar['close']
        assert volume == bar['volume']

        assets = [asset]
        price = broker.get_spot_value(assets, 'price', dt, data_freq)[0]
        last_trade = broker.get_spot_value(
            assets, 'last_traded', dt, data_freq)[0]
        open_ = broker.get_spot_value(assets, 'open', dt, data_freq)[0]
        high = broker.get_spot_value(assets, 'high', dt, data_freq)[0]
        low = broker.get_spot_value(assets, 'low', dt, data_freq)[0]
        close = broker.get_spot_value(assets, 'close', dt, data_freq)[0]
        volume = broker.get_spot_value(assets, 'volume', dt, data_freq)[0]

        assert price == quote['last']
        assert last_trade == pd.Timestamp(quote['last_timestamp'])
        assert open_ == bar['open']
        assert high == bar['high']
        assert low == bar['low']
        assert close == bar['close']
        assert volume == bar['volume']

    @patch('zipline.gens.brokers.alpaca_broker.tradeapi')
    def test_is_alive(self, tradeapi):
        api = tradeapi.REST()
        broker = ALPACABroker('')
        assert broker.is_alive()
        api.get_account.side_effect = Exception()
        assert not broker.is_alive()

    @patch('zipline.gens.brokers.alpaca_broker.symbol_lookup')
    @patch('zipline.gens.brokers.alpaca_broker.tradeapi')
    def test_order(self, tradeapi, symbol_lookup):
        api = tradeapi.REST()
        asset = self.asset_finder.retrieve_asset(1)
        symbol_lookup.return_value = asset
        broker = ALPACABroker('')
        amount = 10

        submitted_orders = []

        def submit_order(symbol, qty, side, type, time_in_force,
                         limit_price, stop_price, client_order_id):
            o = apca.Order({
                'symbol': symbol,
                'qty': str(qty),
                'side': side,
                'type': type,
                'time_in_force': time_in_force,
                'limit_price': limit_price,
                'stop_price': stop_price,
                'client_order_id': client_order_id,
                'submitted_at': '2017-06-01T10:30:00-0400',
                'filled_at': None,
                'filled_qty': None,
                'canceled_at': None,
                'failed_at': None,
            })
            submitted_orders.append(o)
            return o
        api.submit_order = submit_order
        order = broker.order(asset, amount, MarketOrder())
        assert order.limit is None
        assert submitted_orders[-1].side == 'buy'
        assert submitted_orders[-1].type == 'market'

        order = broker.order(asset, -amount, LimitOrder(210.00))
        assert order.limit == 210.00
        assert order.amount == -amount
        assert submitted_orders[-1].side == 'sell'
        assert submitted_orders[-1].type == 'limit'
        assert submitted_orders[-1].limit_price is not None

        order = broker.order(asset, amount, StopOrder(211))
        assert order.stop == 211.00
        assert submitted_orders[-1].side == 'buy'
        assert submitted_orders[-1].type == 'stop'
        assert submitted_orders[-1].stop_price is not None

        order = broker.order(asset, -amount, StopLimitOrder(210, 211))
        assert order.limit == 210.00
        assert order.stop == 211.00
        assert submitted_orders[-1].side == 'sell'
        assert submitted_orders[-1].type == 'stop_limit'
        assert submitted_orders[-1].limit_price is not None
        assert submitted_orders[-1].stop_price is not None

        api.get_order_by_client_order_id.return_value = submitted_orders[-1]

        def cancel_order(self, order_id):
            assert order.id == order_id

        broker.cancel_order(order.id)

    @patch('zipline.gens.brokers.alpaca_broker.symbol_lookup')
    @patch('zipline.gens.brokers.alpaca_broker.tradeapi')
    def test_orders(self, tradeapi, symbol_lookup):
        asset = self.asset_finder.retrieve_asset(1)
        symbol_lookup.return_value = asset
        api = tradeapi.REST()
        id1 = '98486056-5b88-48be-a64c-342e8b751cb2'
        order1 = apca.Order({
            'id': 'order1',
            'symbol': 'SPY',
            'qty': '10',
            'side': 'buy',
            'filled_at': '2017-06-01T10:05:00-0400',
            'submitted_at': '2017-06-01T10:04:30-0400',
            'filled_qty': '10',
            'filled_avg_price': '210.05',
            'failed_at': None,
            'canceled_at': None,
            'limit_price': '210.32',
            'stop_price': '210.00',
            'client_order_id': id1,
        })
        # OPEN
        id2 = 'debb01c4-e40e-4ac0-a6b0-8aed1a7fc126'
        order2 = apca.Order({
            'id': 'order2',
            'symbol': 'SPY',
            'qty': '10',
            'side': 'sell',
            'filled_at': None,
            'submitted_at': '2017-06-01T10:04:30-0400',
            'filled_qty': None,
            'filled_avg_price': None,
            'failed_at': None,
            'canceled_at': None,
            'limit_price': '210.32',
            'stop_price': '210.00',
            'client_order_id': id2,
        })
        # CANCELED
        id3 = 'ac5c7cda-f8ff-4b6f-ad17-fd68f7769ae9'
        order3 = apca.Order({
            'id': 'order3',
            'symbol': 'SPY',
            'qty': '10',
            'side': 'sell',
            'filled_at': None,
            'submitted_at': '2017-06-01T10:04:30-0400',
            'filled_qty': None,
            'filled_avg_price': None,
            'failed_at': None,
            'canceled_at': '2017-06-01T10:04:31-0400',
            'limit_price': '210.32',
            'stop_price': '210.00',
            'client_order_id': id3,
        })
        # REJECTED
        id4 = 'a4018fb1-cc9c-429a-a452-f92e8dc4096b'
        order4 = apca.Order({
            'id': 'order4',
            'symbol': 'SPY',
            'qty': '10',
            'side': 'sell',
            'filled_at': None,
            'submitted_at': '2017-06-01T10:04:30-0400',
            'filled_qty': None,
            'filled_avg_price': None,
            'failed_at': '2017-06-01T10:04:31-0400',
            'canceled_at': None,
            'limit_price': '210.32',
            'stop_price': '210.00',
            'client_order_id': id4,
        })
        api.list_orders.return_value = [
            order1,
            order2,
            order3,
            order4,
        ]
        broker = ALPACABroker('')
        orders = broker.orders
        assert orders[id1].status == ORDER_STATUS.FILLED
        assert orders[id1].filled == int(order1.filled_qty)
        assert orders[id1].amount == int(order1.qty)
        assert orders[id1].asset == asset
        assert orders[id2].status == ORDER_STATUS.OPEN
        assert orders[id3].status == ORDER_STATUS.CANCELLED
        assert orders[id4].status == ORDER_STATUS.REJECTED

        trans = broker.orders
        assert len(trans) == 4

        trans = broker.transactions
        assert len(trans) == 1
        assert trans[id1].amount == 10

    @unittest.skip
    @patch('zipline.gens.brokers.alpaca_broker.symbol_lookup')
    @patch('zipline.gens.brokers.alpaca_broker.tradeapi')
    def test_portfolio(self, tradeapi, symbol_lookup):
        api = tradeapi.REST()
        asset = self.asset_finder.retrieve_asset(1)
        ret_account = apca.Account({
            'cash': '5000.00',
            'portfolio_value': '7000.00'
        })
        api.get_account.return_value = ret_account
        ret_positions = [
            apca.Position({
                'symbol': 'SPY',
                'qty': '10',
                'cost_basis': '210.00',
            })
        ]
        api.list_positions.return_value = ret_positions
        ret_quotes = [
            apca.Quote({
                'symbol': 'SPY',
                'last': 210.05,
                'last_timestamp': '2017-06-01T10:03:03-0400',
            })
        ]
        api.list_quotes.return_value = ret_quotes
        symbol_lookup.return_value = asset
        broker = ALPACABroker('')
        portfolio = broker.portfolio

        assert portfolio.cash == float(ret_account.cash)

        account = broker.account
        assert account.buying_power == float(ret_account.cash)
        assert account.total_position_value == float(
            ret_account.portfolio_value) - float(ret_account.cash)

        positions = portfolio.positions
        assert positions[asset].cost_basis == float(
            ret_positions[0].cost_basis)
        assert positions[asset].last_sale_price == float(ret_quotes[0].last)

        portfolio = broker.portfolio

    @patch('zipline.gens.brokers.alpaca_broker.tradeapi')
    def test_last_trade_dt(self, tradeapi):
        asset = self.asset_finder.retrieve_asset(1)
        api = tradeapi.REST()
        broker = ALPACABroker('')

        def get_quote(symbol):
            assert symbol == 'SPY'
            return apca.Quote({
                'symbol': 'SPY',
                'last': 210,
                'last_timestamp': '2016-06-01T10:27:00-0400',
            })
        api.get_quote = get_quote
        ret = broker.get_last_traded_dt(asset)
        assert ret.minute == 27

    @patch('zipline.gens.brokers.alpaca_broker.tradeapi')
    def test_misc(self, tradeapi):
        broker = ALPACABroker('')
        assert broker.subscribe_to_market_data(None) is None
        assert broker.subscribed_assets() == []
        assert broker.time_skew == pd.Timedelta('0sec')
