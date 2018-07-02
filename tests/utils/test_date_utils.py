from pandas import Timestamp
from nose_parameterized import parameterized
from trading_calendars import get_calendar

from zipline.testing import ZiplineTestCase
from zipline.utils.date_utils import compute_date_range_chunks


def T(s):
    """
    Helpful function to improve readibility.
    """
    return Timestamp(s, tz='UTC')


class TestDateUtils(ZiplineTestCase):

    @classmethod
    def init_class_fixtures(cls):
        super(TestDateUtils, cls).init_class_fixtures()
        cls.calendar = get_calendar('NYSE')

    @parameterized.expand([
        (None, [(T('2017-01-03'), T('2017-01-31'))]),
        (10, [
            (T('2017-01-03'), T('2017-01-17')),
            (T('2017-01-18'), T('2017-01-31'))
        ]),
        (15, [
            (T('2017-01-03'), T('2017-01-24')),
            (T('2017-01-25'), T('2017-01-31'))
        ]),
    ])
    def test_compute_date_range_chunks(self, chunksize, expected):
        # This date range results in 20 business days
        start_date = T('2017-01-03')
        end_date = T('2017-01-31')

        date_ranges = compute_date_range_chunks(
            self.calendar.all_sessions,
            start_date,
            end_date,
            chunksize
        )

        self.assertListEqual(list(date_ranges), expected)

    def test_compute_date_range_chunks_invalid_input(self):
        # Start date not found in calendar
        with self.assertRaises(KeyError) as cm:
            compute_date_range_chunks(
                self.calendar.all_sessions,
                T('2017-05-07'),  # Sunday
                T('2017-06-01'),
                None
            )
        self.assertEqual(
            str(cm.exception),
            "'Start date 2017-05-07 is not found in calendar.'"
        )

        # End date not found in calendar
        with self.assertRaises(KeyError) as cm:
            compute_date_range_chunks(
                self.calendar.all_sessions,
                T('2017-05-01'),
                T('2017-05-27'),  # Saturday
                None
            )
        self.assertEqual(
            str(cm.exception),
            "'End date 2017-05-27 is not found in calendar.'"
        )

        # End date before start date
        with self.assertRaises(ValueError) as cm:
            compute_date_range_chunks(
                self.calendar.all_sessions,
                T('2017-06-01'),
                T('2017-05-01'),
                None
            )
        self.assertEqual(
            str(cm.exception),
            "End date 2017-05-01 cannot precede start date 2017-06-01."
        )
