import datetime

import pytz

from unittest import TestCase

from zipline.finance.cancel_policy import NeverCancel, EODCancel
from zipline.finance.order import Order
from zipline.gens.sim_engine import (
    BAR,
    DAY_END
)


class CancelPolicyTestCase(TestCase):

    def test_eod_cancel(self):

        order = Order(
            dt=datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
            sid=133,
            amount=1,
            cancel_policy=EODCancel()
        )

        current_dt = datetime.datetime(2006, 1, 5, 20, 59, tzinfo=pytz.utc)
        current_event = BAR

        should_cancel = order.should_cancel(current_dt, current_event)
        self.assertEqual(should_cancel, False)

        current_dt = datetime.datetime(2006, 1, 5, 21, 0, tzinfo=pytz.utc)
        current_event = DAY_END

        should_cancel = order.should_cancel(current_dt, current_event)
        self.assertEqual(should_cancel, True)

    def test_never_cancel(self):

        order = Order(
            dt=datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
            sid=133,
            amount=1,
            cancel_policy=NeverCancel()
        )

        current_dt = datetime.datetime(2006, 1, 5, 20, 59, tzinfo=pytz.utc)
        current_event = BAR

        should_cancel = order.should_cancel(current_dt, current_event)
        self.assertEqual(should_cancel, False)

        current_dt = datetime.datetime(2006, 1, 5, 21, 0, tzinfo=pytz.utc)
        current_event = DAY_END

        should_cancel = order.should_cancel(current_dt, current_event)
        self.assertEqual(should_cancel, False)
