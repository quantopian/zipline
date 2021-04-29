import pandas as pd
from trading_calendars import get_calendar

from zipline.utils.date_utils import compute_date_range_chunks
import pytest


def T(s):
    """
    Helpful function to improve readibility.
    """
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture(scope="class")
def set_calendar(request):
    request.cls.calendar = get_calendar("XNYS")


@pytest.mark.usefixtures("set_calendar")
class TestDateUtils:
    @pytest.mark.parametrize(
        "chunksize, expected",
        [
            (None, [(T("2017-01-03"), T("2017-01-31"))]),
            (
                10,
                [
                    (T("2017-01-03"), T("2017-01-17")),
                    (T("2017-01-18"), T("2017-01-31")),
                ],
            ),
            (
                15,
                [
                    (T("2017-01-03"), T("2017-01-24")),
                    (T("2017-01-25"), T("2017-01-31")),
                ],
            ),
        ],
    )
    def test_compute_date_range_chunks(self, chunksize, expected):
        # This date range results in 20 business days
        start_date = T("2017-01-03")
        end_date = T("2017-01-31")

        date_ranges = compute_date_range_chunks(
            self.calendar.all_sessions, start_date, end_date, chunksize
        )

        assert list(date_ranges) == expected

    def test_compute_date_range_chunks_invalid_input(self):
        # Start date not found in calendar
        err_msg = "'Start date 2017-05-07 is not found in calendar.'"
        with pytest.raises(KeyError, match=err_msg):
            compute_date_range_chunks(
                self.calendar.all_sessions,
                T("2017-05-07"),  # Sunday
                T("2017-06-01"),
                None,
            )

        # End date not found in calendar
        err_msg = "'End date 2017-05-27 is not found in calendar.'"
        with pytest.raises(KeyError, match=err_msg):
            compute_date_range_chunks(
                self.calendar.all_sessions,
                T("2017-05-01"),
                T("2017-05-27"),  # Saturday
                None,
            )

        # End date before start date
        err_msg = "End date 2017-05-01 cannot precede start date 2017-06-01."
        with pytest.raises(ValueError, match=err_msg):
            compute_date_range_chunks(
                self.calendar.all_sessions, T("2017-06-01"), T("2017-05-01"), None
            )
