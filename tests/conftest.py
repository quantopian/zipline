import warnings
import pandas as pd
import pytest
from zipline.utils.calendar_utils import get_calendar
import sqlalchemy as sa

from zipline.assets import (
    AssetDBWriter,
    AssetFinder,
    Equity,
    Future,
)


DEFAULT_DATE_BOUNDS = {
    "START_DATE": pd.Timestamp("2006-01-03"),
    "END_DATE": pd.Timestamp("2006-12-29"),
}


@pytest.fixture(scope="function")
def sql_db(request):
    url = "sqlite:///:memory:"
    request.cls.engine = sa.create_engine(url)
    yield request.cls.engine
    request.cls.engine.dispose()
    request.cls.engine = None


@pytest.fixture(scope="class")
def sql_db_class(request):
    url = "sqlite:///:memory:"
    request.cls.engine = sa.create_engine(url)
    yield request.cls.engine
    request.cls.engine.dispose()
    request.cls.engine = None


@pytest.fixture(scope="function")
def empty_assets_db(sql_db, request):
    AssetDBWriter(sql_db).write(None)
    request.cls.metadata = sa.MetaData()
    request.cls.metadata.reflect(bind=sql_db)


@pytest.fixture(scope="class")
def with_trading_calendars(request):
    """fixture providing cls.trading_calendar,
    cls.all_trading_calendars, cls.trading_calendar_for_asset_type as a
    class-level fixture.

    - `cls.trading_calendar` is populated with a default of the nyse trading
    calendar for compatibility with existing tests
    - `cls.all_trading_calendars` is populated with the trading calendars
    keyed by name,
    - `cls.trading_calendar_for_asset_type` is populated with the trading
    calendars keyed by the asset type which uses the respective calendar.

    Attributes
    ----------
    TRADING_CALENDAR_STRS : iterable
        iterable of identifiers of the calendars to use.
    TRADING_CALENDAR_FOR_ASSET_TYPE : dict
        A dictionary which maps asset type names to the calendar associated
        with that asset type.
    """

    request.cls.TRADING_CALENDAR_STRS = ("NYSE",)
    request.cls.TRADING_CALENDAR_FOR_ASSET_TYPE = {Equity: "NYSE", Future: "us_futures"}
    # For backwards compatibility, exisitng tests and fixtures refer to
    # `trading_calendar` with the assumption that the value is the NYSE
    # calendar.
    request.cls.TRADING_CALENDAR_PRIMARY_CAL = "NYSE"

    request.cls.trading_calendars = {}
    # Silence `pandas.errors.PerformanceWarning: Non-vectorized DateOffset
    # being applied to Series or DatetimeIndex` in trading calendar
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for cal_str in set(request.cls.TRADING_CALENDAR_STRS) | {
            request.cls.TRADING_CALENDAR_PRIMARY_CAL
        }:
            # Set name to allow aliasing.
            calendar = get_calendar(cal_str)
            setattr(request.cls, "{0}_calendar".format(cal_str.lower()), calendar)
            request.cls.trading_calendars[cal_str] = calendar

        type_to_cal = request.cls.TRADING_CALENDAR_FOR_ASSET_TYPE.items()
        for asset_type, cal_str in type_to_cal:
            calendar = get_calendar(cal_str)
            request.cls.trading_calendars[asset_type] = calendar

    request.cls.trading_calendar = request.cls.trading_calendars[
        request.cls.TRADING_CALENDAR_PRIMARY_CAL
    ]


@pytest.fixture(scope="class")
def set_trading_calendar():
    TRADING_CALENDAR_STRS = ("NYSE",)
    TRADING_CALENDAR_FOR_ASSET_TYPE = {Equity: "NYSE", Future: "us_futures"}
    # For backwards compatibility, exisitng tests and fixtures refer to
    # `trading_calendar` with the assumption that the value is the NYSE
    # calendar.
    TRADING_CALENDAR_PRIMARY_CAL = "NYSE"

    trading_calendars = {}
    # Silence `pandas.errors.PerformanceWarning: Non-vectorized DateOffset
    # being applied to Series or DatetimeIndex` in trading calendar
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for cal_str in set(TRADING_CALENDAR_STRS) | {TRADING_CALENDAR_PRIMARY_CAL}:
            # Set name to allow aliasing.
            calendar = get_calendar(cal_str)
            # setattr(request.cls, "{0}_calendar".format(cal_str.lower()), calendar)
            trading_calendars[cal_str] = calendar

        type_to_cal = TRADING_CALENDAR_FOR_ASSET_TYPE.items()
        for asset_type, cal_str in type_to_cal:
            calendar = get_calendar(cal_str)
            trading_calendars[asset_type] = calendar

    return trading_calendars[TRADING_CALENDAR_PRIMARY_CAL]


@pytest.fixture(scope="class")
def with_asset_finder(sql_db_class):
    def asset_finder(**kwargs):
        AssetDBWriter(sql_db_class).write(**kwargs)
        return AssetFinder(sql_db_class)

    return asset_finder


@pytest.fixture(scope="class")
def with_benchmark_returns(request):
    from zipline.testing.fixtures import (
        read_checked_in_benchmark_data,
        STATIC_BENCHMARK_PATH,
    )

    START_DATE = DEFAULT_DATE_BOUNDS["START_DATE"].date()
    END_DATE = DEFAULT_DATE_BOUNDS["END_DATE"].date()

    benchmark_returns = read_checked_in_benchmark_data()

    # Zipline ordinarily uses cached benchmark returns data, but when
    # running the zipline tests this cache is not always updated to include
    # the appropriate dates required by both the futures and equity
    # calendars. In order to create more reliable and consistent data
    # throughout the entirety of the tests, we read static benchmark
    # returns files from source. If a test using this fixture attempts to
    # run outside of the static date range of the csv files, raise an
    # exception warning the user to either update the csv files in source
    # or to use a date range within the current bounds.
    static_start_date = benchmark_returns.index[0].date()
    static_end_date = benchmark_returns.index[-1].date()
    warning_message = (
        "The WithBenchmarkReturns fixture uses static data between "
        "{static_start} and {static_end}. To use a start and end date "
        "of {given_start} and {given_end} you will have to update the "
        "file in {benchmark_path} to include the missing dates.".format(
            static_start=static_start_date,
            static_end=static_end_date,
            given_start=START_DATE,
            given_end=END_DATE,
            benchmark_path=STATIC_BENCHMARK_PATH,
        )
    )
    if START_DATE < static_start_date or END_DATE > static_end_date:
        raise AssertionError(warning_message)

    request.cls.BENCHMARK_RETURNS = benchmark_returns
