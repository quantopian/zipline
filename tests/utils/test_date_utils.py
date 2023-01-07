import pandas as pd
from zipline.utils.calendar_utils import get_calendar
from pandas.testing import assert_index_equal

from zipline.utils.date_utils import compute_date_range_chunks, make_utc_aware
import pytest


def T(s, tz=None):
    """Helpful function to improve readability."""
    return pd.Timestamp(s, tz=tz)


def DTI(start=None, end=None, periods=None, freq=None, tz=None, normalize=False):
    """Creates DateTimeIndex using pd.date_range."""
    return pd.date_range(start, end, periods, freq, tz, normalize)


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
        start_date = pd.Timestamp("2017-01-03")
        end_date = pd.Timestamp("2017-01-31")

        date_ranges = compute_date_range_chunks(
            self.calendar.sessions, start_date, end_date, chunksize
        )

        assert list(date_ranges) == expected

    def test_compute_date_range_chunks_invalid_input(self):
        # Start date not found in calendar
        err_msg = "'Start date 2017-05-07 is not found in calendar.'"
        with pytest.raises(KeyError, match=err_msg):
            compute_date_range_chunks(
                self.calendar.sessions,
                T("2017-05-07"),  # Sunday
                T("2017-06-01"),
                None,
            )

        # End date not found in calendar
        err_msg = "'End date 2017-05-27 is not found in calendar.'"
        with pytest.raises(KeyError, match=err_msg):
            compute_date_range_chunks(
                self.calendar.sessions,
                T("2017-05-01"),
                T("2017-05-27"),  # Saturday
                None,
            )

        # End date before start date
        err_msg = "End date 2017-05-01 cannot precede start date 2017-06-01."
        with pytest.raises(ValueError, match=err_msg):
            compute_date_range_chunks(
                self.calendar.sessions, T("2017-06-01"), T("2017-05-01"), None
            )


class TestMakeTZAware:
    @pytest.mark.parametrize(
        "dti, expected",
        [
            (
                DTI(start="2020-01-01", end="2020-02-01"),
                DTI(start="2020-01-01", end="2020-02-01", tz=None).tz_localize("UTC"),
            ),
            (
                DTI(start="2020-01-01", end="2020-02-01", tz="UTC"),
                DTI(start="2020-01-01", end="2020-02-01", tz="UTC"),
            ),
            (
                DTI(start="2020-01-01", end="2020-02-01", tz="US/Eastern"),
                DTI(start="2020-01-01", end="2020-02-01", tz="US/Eastern").tz_convert(
                    "UTC"
                ),
            ),
        ],
    )
    def test_index_converts(self, dti, expected):
        # GIVEN a pd.DateTimeIndex (DTI)
        # WHEN it has NO/UTC/other TZ info
        # THEN returned DTI has UTC tz_info
        result = make_utc_aware(dti=dti)
        assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "ts, expected",
        [
            (T("2020-01-01"), T("2020-01-01", tz=None).tz_localize("UTC")),
            (T("2020-01-01", tz="UTC"), T("2020-01-01", tz="UTC")),
            (
                T("2020-01-01", tz="US/Eastern"),
                T("2020-01-01", tz="US/Eastern").tz_convert("UTC"),
            ),
        ],
    )
    def test_time_stamp_converts(self, ts, expected):
        # GIVEN a pd.TimeStamp (DTI)
        # WHEN it has NO/UTC/other TZ info
        # THEN returned DTI has UTC tz_info
        result = make_utc_aware(dti=ts)
        assert result == expected
