import pandas as pd

# fix to allow zip_longest on Python 2.X and 3.X
try:                                    # Python 3
    from itertools import zip_longest
except ImportError:                     # Python 2
    from itertools import izip_longest as zip_longest

from mock import patch, sentinel, Mock, MagicMock
from ib.ext.Execution import Execution
from zipline.finance.order import Order as ZPOrder
from zipline.finance.blotter.blotter_live import BlotterLive
from zipline.gens.brokers.broker import Broker
from zipline.testing.fixtures import WithSimParams
from zipline.finance.transaction import Transaction
from zipline.testing.fixtures import (ZiplineTestCase,
                                      WithDataPortal)

class TestBlotterLive(WithSimParams,
                      WithDataPortal,
                      ZiplineTestCase):
    ASSET_FINDER_EQUITY_SIDS = (1, 2)
    ASSET_FINDER_EQUITY_SYMBOLS = ("SPY", "XIV")

    @staticmethod
    def _get_orders(asset1, asset2):
        return {
            sentinel.order_id1: ZPOrder(
                dt=sentinel.dt, asset=asset1, amount=12, commission=12,
                stop=sentinel.stop1, limit=sentinel.limit1,
                id=sentinel.order_id1),
            sentinel.order_id2: ZPOrder(
                dt=sentinel.dt, asset=asset1, amount=-12, commission=12,
                limit=sentinel.limit2, id=sentinel.order_id2),
            sentinel.order_id3: ZPOrder(
                dt=sentinel.dt, asset=asset2, amount=3, commission=3,
                stop=sentinel.stop2, limit=sentinel.limit2,
                id=sentinel.order_id3),
            sentinel.order_id4: ZPOrder(
                dt=sentinel.dt, asset=asset2, amount=-122, commission=122,
                id=sentinel.order_id4),
        }

    @staticmethod
    def _get_execution(price, qty, dt):
        execution = Execution()
        execution.m_price = price
        execution.m_cumQty = qty
        execution.m_time = dt
        return execution

    def test_open_orders(self):
        broker = MagicMock(Broker)
        blotter = BlotterLive(data_frequency='minute', broker=broker)
        assert not blotter.open_orders

        asset1 = self.asset_finder.retrieve_asset(1)
        asset2 = self.asset_finder.retrieve_asset(2)

        all_orders = self._get_orders(asset1, asset2)
        all_orders[sentinel.order_id4].filled = -122
        broker.orders = all_orders

        assert len(blotter.open_orders) == 2

        assert len(blotter.open_orders[asset1]) == 2
        assert all_orders[sentinel.order_id1] in blotter.open_orders[asset1]
        assert all_orders[sentinel.order_id2] in blotter.open_orders[asset1]

        assert len(blotter.open_orders[asset2]) == 1
        assert blotter.open_orders[asset2][0].id == sentinel.order_id3

    def test_get_transactions(self):
        broker = MagicMock(Broker)
        blotter = BlotterLive(data_frequency='minute', broker=broker)

        asset1 = self.asset_finder.retrieve_asset(1)
        asset2 = self.asset_finder.retrieve_asset(2)

        broker.orders = {}
        broker.transactions = {}
        new_transactions, new_commissions, new_closed_orders = \
            blotter.get_transactions(None)
        assert not new_transactions
        assert not new_commissions
        assert not new_closed_orders

        broker.orders = self._get_orders(asset1, asset2)
        new_transactions, new_commissions, new_closed_orders = \
            blotter.get_transactions(None)
        assert not new_transactions
        assert not new_commissions
        assert not new_closed_orders

        broker.orders[sentinel.order_id4].filled = \
            broker.orders[sentinel.order_id4].amount
        broker.transactions['exec_4'] = \
            Transaction(asset=asset2,
                        amount=broker.orders[sentinel.order_id4].amount,
                        dt=pd.to_datetime('now', utc=True),
                        price=123, order_id=sentinel.order_id4)
        new_transactions, new_commissions, new_closed_orders = \
            blotter.get_transactions(None)
        assert new_closed_orders == [broker.orders[sentinel.order_id4], ]
        assert new_commissions == [{
            'asset': asset2,
            'cost': 122,
            'order': broker.orders[sentinel.order_id4]
        }]
        assert new_transactions == [list(broker.transactions.values())[0], ]

        new_transactions, new_commissions, new_closed_orders = \
            blotter.get_transactions(None)
        assert not new_transactions
        assert not new_commissions
        assert not new_closed_orders

        broker.orders[sentinel.order_id3].filled = \
            broker.orders[sentinel.order_id3].amount
        broker.transactions['exec_3'] = \
            Transaction(asset=asset1,
                        amount=broker.orders[sentinel.order_id3].amount,
                        dt=pd.to_datetime('now', utc=True),
                        price=1234, order_id=sentinel.order_id3)

        broker.orders[sentinel.order_id2].filled = \
            broker.orders[sentinel.order_id2].amount
        broker.transactions['exec_2'] = \
            Transaction(asset=asset2,
                        amount=broker.orders[sentinel.order_id2].amount,
                        dt=pd.to_datetime('now', utc=True),
                        price=12.34, order_id=sentinel.order_id2)

        new_transactions, new_commissions, new_closed_orders = \
            blotter.get_transactions(None)
        assert len(new_closed_orders) == 2
        assert broker.orders[sentinel.order_id3] in new_closed_orders
        assert broker.orders[sentinel.order_id2] in new_closed_orders

        assert len(new_commissions) == 2
        assert {'asset': asset2,
                'cost': 12,
                'order': broker.orders[sentinel.order_id2]}\
            in new_commissions
        assert {'asset': asset1,
                'cost': 3,
                'order': broker.orders[sentinel.order_id3]} \
            in new_commissions
        assert len(new_transactions) == 2
        assert broker.transactions['exec_2'] in new_transactions
        assert broker.transactions['exec_3'] in new_transactions

        new_transactions, new_commissions, new_closed_orders = \
            blotter.get_transactions(None)
        assert not new_transactions
        assert not new_commissions
        assert not new_closed_orders
