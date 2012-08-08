import pytz

from datetime import timedelta, datetime
from collections import defaultdict
from unittest2 import TestCase

from zipline import ndict

from zipline.lines import SimulatedTrading

from zipline.utils.test_utils import setup_logger, teardown_logger
from zipline.utils.date_utils import utcnow

from zipline.gens.tradegens import SpecificEquityTrades
from zipline.gens.transform import StatefulTransform, EventWindow
from zipline.gens.vwap import VWAP
from zipline.gens.mavg import MovingAverage
from zipline.gens.returns import Returns

import zipline.utils.factory as factory

def to_dt(msg):
    return ndict({'dt': msg})

class NoopEventWindow(EventWindow):
    """
    A no-op EventWindow subclass for testing the base EventWindow logic.
    Keeps lists of all added and dropped events.
    """
    def __init__(self, market_aware, days, delta):
        EventWindow.__init__(self, market_aware, days, delta)

        self.added = []
        self.removed = []

    def handle_add(self, event):
        self.added.append(event)

    def handle_remove(self, event):
        self.removed.append(event)

class EventWindowTestCase(TestCase):

    def setUp(self):
        setup_logger(self)
        
        # Constants calling before open, during the day, and after
        # close on a valid trading day.
        self.pre_open = datetime(2012, 8, 7, 13, tzinfo = pytz.utc)
        self.mid_day = datetime(2012, 8, 7, 15, tzinfo = pytz.utc)
        self.post_close = datetime(2012, 8, 7, 22, tzinfo = pytz.utc)

        # Constants calling before open, during the day, and after
        # close on a saturday.
        self.pre_open_saturday = datetime(2012, 8, 11, 13, tzinfo = pytz.utc)
        self.mid_day_saturday = datetime(2012, 8, 11, 15, tzinfo = pytz.utc)
        self.post_close_saturday = datetime(2012, 8, 11, 22, tzinfo = pytz.utc)

        # Constants calling before open, during the day, and after
        # close on a holiday.
        self.pre_open_holiday = datetime(2012, 12, 25, 13, tzinfo = pytz.utc)
        self.mid_day_holiday = datetime(2012, 12, 25, tzinfo = pytz.utc)
        self.post_close_holiday = datetime(2012, 12, 25, 22, tzinfo = pytz.utc)
                          
    def test_event_window_with_timedelta(self):
        
        # Keep all events within a 5 minute window.
        window = NoopEventWindow(
            market_aware = False, 
            delta = timedelta(minutes = 5),
            days = None
        )
        now = utcnow()

        # 15 dates, increasing in 1 minute increments.
        dates = [now + i * timedelta(minutes = 1)
                 for i in xrange(15)]

        # Turn the dates into the format required by EventWindow.
        dt_messages = [to_dt(date) for date in dates]

        # Run all messages through the window and assert that we're adding
        # and removing messages appropriately. We start the enumeration at 1
        # for convenience.
        for num, message in enumerate(dt_messages, 1):
            window.update(message)

            # Assert that we've added the correct number of events.
            assert len(window.added) == num

            # Assert that we removed only events that fall outside (or
            # on the boundary of) the delta.
            for dropped in window.removed:
                assert message.dt - dropped.dt >= timedelta(minutes = 5)

    def test_market_aware_window(self):
        window = NoopEventWindow(
            market_aware = True, 
            delta = None,
            days = 1
        )
        dates =  ([self.pre_open]*3)
        dates += ([self.mid_day]*3)
        dates += ([self.post_close]*3)
        dates += [self.pre_open + timedelta(days = 1, seconds = 1)]
        events = [to_dt(date) for date in dates]

        # Run the events.
        for event in events:
            window.update(event)
        
        # We should have removed the pre_open events on the first day.
        # The rest should be intact.
            
        assert window.added == events
        assert window.removed == events[0:3]
        assert list(window.ticks) == events[3:]

    def test_market_aware_window_weekend(self):
        window = NoopEventWindow(
            market_aware = True, 
            delta = None,
            days = 2
        )
        dates = [self.pre_open_saturday - timedelta(days = 1, seconds=1)]
        dates += [self.mid_day_saturday - timedelta(days = 1, seconds=1)]
        dates += [self.post_close_saturday - timedelta(days = 1, seconds=1)]
        dates += [self.mid_day_saturday + timedelta(days = 1)]
        
        events = [to_dt(date) for date in dates]

        # Run the events.
        for event in events:
            window.update(event)
        
        # We shouldn't remove any events.
        assert window.added == events
        assert window.removed == []
        assert list(window.ticks) == events
        
        extra = to_dt(self.mid_day_saturday + timedelta(days = 2))
        window.update(extra)

        # We should remove only the first event.
        assert window.removed == [events[0]]
        assert list(window.ticks) == events[1:] + [extra]
        
    def tearDown(self):
        setup_logger(self)

class FinanceTransformsTestCase(TestCase):

    def setUp(self):
        self.trading_environment = factory.create_trading_environment()
        setup_logger(self)

        trade_history = factory.create_trade_history(
            133,
            [10.0, 10.0, 11.0, 11.0],
            [100, 100, 100, 300],
            timedelta(days=1),
            self.trading_environment
        )
        self.source = SpecificEquityTrades(event_list=trade_history)

    def tearDown(self):
        self.log_handler.pop_application()

    def test_vwap(self):

        vwap = StatefulTransform(
            VWAP,
            market_aware = False,
            delta = timedelta(days = 2)
        )
        transformed = list(vwap.transform(self.source))

        # Output values
        tnfm_vals = [message.tnfm_value for message in transformed]
        # "Hand calculated" values.
        expected = [
            (10.0 * 100) / 100.0,
            ((10.0 * 100) + (10.0 * 100)) / (200.0),
            # We should drop the first event here.
            ((10.0 * 100) + (11.0 * 100)) / (200.0),
            # We should drop the second event here.
            ((11.0 * 100) + (11.0 * 300)) / (400.0)
        ]

        # Output should match the expected.
        assert tnfm_vals == expected

    def test_returns(self):
        # Daily returns.
        returns = StatefulTransform(Returns, 1)

        transformed = list(returns.transform(self.source))
        tnfm_vals = [message.tnfm_value for message in transformed]

        # No returns for the first event because we don't have a
        # previous close.
        expected = [0.0, 0.0, 0.1, 0.0]

        assert tnfm_vals == expected
        
        # Two-day returns.  An extra kink here is that the
        # factory will automatically skip a weekend for the
        # last event. Results shouldn't notice this blip.

        trade_history = factory.create_trade_history(
            133,
            [10.0, 15.0, 13.0, 12.0, 13.0],
            [100, 100, 100, 300, 100],
            timedelta(days=1),
            self.trading_environment
        )
        self.source = SpecificEquityTrades(event_list=trade_history)

        returns = StatefulTransform(Returns, 2)

        transformed = list(returns.transform(self.source))
        tnfm_vals = [message.tnfm_value for message in transformed]

        expected = [
            0.0,
            0.0,
            (13.0 - 10.0) / 10.0,
            (12.0 - 15.0) / 15.0,
            (13.0 - 13.0) / 13.0
        ]

        assert tnfm_vals == expected

    def test_moving_average(self):

        mavg = StatefulTransform(
            MovingAverage,
            market_aware = False,
            fields = ['price', 'volume'],
            delta = timedelta(days = 2),
        )
        transformed = list(mavg.transform(self.source))
        # Output values.
        tnfm_prices = [message.tnfm_value.price for message in transformed]
        tnfm_volumes = [message.tnfm_value.volume for message in transformed]

        # "Hand-calculated" values
        expected_prices = [
            ((10.0) / 1.0),
            ((10.0 + 10.0) / 2.0),
            # First event should get dropped here.
            ((10.0 + 11.0) / 2.0),
            # Second event should get dropped here.
            ((11.0 + 11.0) / 2.0)
        ]
        expected_volumes = [
            ((100.0) / 1.0),
            ((100.0 + 100.0) / 2.0),
            # First event should get dropped here.
            ((100.0 + 100.0) / 2.0),
            # Second event should get dropped here.
            ((100.0 + 300.0) / 2.0)
        ]

        assert tnfm_prices == expected_prices
        assert tnfm_volumes == expected_volumes
