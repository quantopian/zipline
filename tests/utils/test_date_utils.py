from pandas import Timestamp

from nose_parameterized import parameterized

from zipline.testing import ZiplineTestCase
from zipline.utils.calendars import get_calendar
from zipline.utils.date_utils import roll_dates_to_previous_session


class TestRollDatesToPreviousSession(ZiplineTestCase):

    @parameterized.expand([
        (
            Timestamp('05-19-2017', tz='UTC'),  # actual trading date
            Timestamp('05-19-2017', tz='UTC'),
        ),
        (
            Timestamp('07-04-2015', tz='UTC'),  # weekend nyse holiday
            Timestamp('07-02-2015', tz='UTC'),
        ),
        (
            Timestamp('01-16-2017', tz='UTC'),  # weeknight nyse holiday
            Timestamp('01-13-2017', tz='UTC'),
        ),
    ])
    def test_roll_dates_to_previous_session(self, date, expected_rolled_date):
        calendar = get_calendar('NYSE')
        result = roll_dates_to_previous_session(calendar, date)
        self.assertEqual(result[0], expected_rolled_date)
