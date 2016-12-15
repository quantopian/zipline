from itertools import product
from string import ascii_uppercase

import pandas as pd
from pandas.tseries.offsets import MonthBegin
from six import iteritems

from .futures import CME_CODE_TO_MONTH


def make_rotating_equity_info(num_assets,
                              first_start,
                              frequency,
                              periods_between_starts,
                              asset_lifetime):
    """
    Create a DataFrame representing lifetimes of assets that are constantly
    rotating in and out of existence.

    Parameters
    ----------
    num_assets : int
        How many assets to create.
    first_start : pd.Timestamp
        The start date for the first asset.
    frequency : str or pd.tseries.offsets.Offset (e.g. trading_day)
        Frequency used to interpret next two arguments.
    periods_between_starts : int
        Create a new asset every `frequency` * `periods_between_new`
    asset_lifetime : int
        Each asset exists for `frequency` * `asset_lifetime` days.

    Returns
    -------
    info : pd.DataFrame
        DataFrame representing newly-created assets.
    """
    return pd.DataFrame(
        {
            'symbol': [chr(ord('A') + i) for i in range(num_assets)],
            # Start a new asset every `periods_between_starts` days.
            'start_date': pd.date_range(
                first_start,
                freq=(periods_between_starts * frequency),
                periods=num_assets,
            ),
            # Each asset lasts for `asset_lifetime` days.
            'end_date': pd.date_range(
                first_start + (asset_lifetime * frequency),
                freq=(periods_between_starts * frequency),
                periods=num_assets,
            ),
            'exchange': 'TEST',
            'exchange_full': 'TEST FULL',
        },
        index=range(num_assets),
    )


def make_simple_equity_info(sids,
                            start_date,
                            end_date,
                            symbols=None):
    """
    Create a DataFrame representing assets that exist for the full duration
    between `start_date` and `end_date`.

    Parameters
    ----------
    sids : array-like of int
    start_date : pd.Timestamp, optional
    end_date : pd.Timestamp, optional
    symbols : list, optional
        Symbols to use for the assets.
        If not provided, symbols are generated from the sequence 'A', 'B', ...

    Returns
    -------
    info : pd.DataFrame
        DataFrame representing newly-created assets.
    """
    num_assets = len(sids)
    if symbols is None:
        symbols = list(ascii_uppercase[:num_assets])
    return pd.DataFrame(
        {
            'symbol': list(symbols),
            'start_date': pd.to_datetime([start_date] * num_assets),
            'end_date': pd.to_datetime([end_date] * num_assets),
            'exchange': 'TEST',
            'exchange_full': 'TEST FULL',
        },
        index=sids,
        columns=(
            'start_date',
            'end_date',
            'symbol',
            'exchange',
            'exchange_full',
        ),
    )


def make_jagged_equity_info(num_assets,
                            start_date,
                            first_end,
                            frequency,
                            periods_between_ends,
                            auto_close_delta):
    """
    Create a DataFrame representing assets that all begin at the same start
    date, but have cascading end dates.

    Parameters
    ----------
    num_assets : int
        How many assets to create.
    start_date : pd.Timestamp
        The start date for all the assets.
    first_end : pd.Timestamp
        The date at which the first equity will end.
    frequency : str or pd.tseries.offsets.Offset (e.g. trading_day)
        Frequency used to interpret the next argument.
    periods_between_ends : int
        Starting after the first end date, end each asset every
        `frequency` * `periods_between_ends`.

    Returns
    -------
    info : pd.DataFrame
        DataFrame representing newly-created assets.
    """
    frame = pd.DataFrame(
        {
            'symbol': [chr(ord('A') + i) for i in range(num_assets)],
            'start_date': start_date,
            'end_date': pd.date_range(
                first_end,
                freq=(periods_between_ends * frequency),
                periods=num_assets,
            ),
            'exchange': 'TEST',
            'exchange_full': 'TEST FULL',
        },
        index=range(num_assets),
    )

    # Explicitly pass None to disable setting the auto_close_date column.
    if auto_close_delta is not None:
        frame['auto_close_date'] = frame['end_date'] + auto_close_delta

    return frame


def make_future_info(first_sid,
                     root_symbols,
                     years,
                     notice_date_func,
                     expiration_date_func,
                     start_date_func,
                     month_codes=None):
    """
    Create a DataFrame representing futures for `root_symbols` during `year`.

    Generates a contract per triple of (symbol, year, month) supplied to
    `root_symbols`, `years`, and `month_codes`.

    Parameters
    ----------
    first_sid : int
        The first sid to use for assigning sids to the created contracts.
    root_symbols : list[str]
        A list of root symbols for which to create futures.
    years : list[int or str]
        Years (e.g. 2014), for which to produce individual contracts.
    notice_date_func : (Timestamp) -> Timestamp
        Function to generate notice dates from first of the month associated
        with asset month code.  Return NaT to simulate futures with no notice
        date.
    expiration_date_func : (Timestamp) -> Timestamp
        Function to generate expiration dates from first of the month
        associated with asset month code.
    start_date_func : (Timestamp) -> Timestamp, optional
        Function to generate start dates from first of the month associated
        with each asset month code.  Defaults to a start_date one year prior
        to the month_code date.
    month_codes : dict[str -> [1..12]], optional
        Dictionary of month codes for which to create contracts.  Entries
        should be strings mapped to values from 1 (January) to 12 (December).
        Default is zipline.futures.CME_CODE_TO_MONTH

    Returns
    -------
    futures_info : pd.DataFrame
        DataFrame of futures data suitable for passing to an AssetDBWriter.
    """
    if month_codes is None:
        month_codes = CME_CODE_TO_MONTH

    year_strs = list(map(str, years))
    years = [pd.Timestamp(s, tz='UTC') for s in year_strs]

    # Pairs of string/date like ('K06', 2006-05-01)
    contract_suffix_to_beginning_of_month = tuple(
        (month_code + year_str[-2:], year + MonthBegin(month_num))
        for ((year, year_str), (month_code, month_num))
        in product(
            zip(years, year_strs),
            iteritems(month_codes),
        )
    )

    contracts = []
    parts = product(root_symbols, contract_suffix_to_beginning_of_month)
    for sid, (root_sym, (suffix, month_begin)) in enumerate(parts, first_sid):
        contracts.append({
            'sid': sid,
            'root_symbol': root_sym,
            'symbol': root_sym + suffix,
            'start_date': start_date_func(month_begin),
            'notice_date': notice_date_func(month_begin),
            'expiration_date': notice_date_func(month_begin),
            'multiplier': 500,
            'exchange': "TEST",
            'exchange_full': 'TEST FULL',
        })
    return pd.DataFrame.from_records(contracts, index='sid')


def make_commodity_future_info(first_sid,
                               root_symbols,
                               years,
                               month_codes=None):
    """
    Make futures testing data that simulates the notice/expiration date
    behavior of physical commodities like oil.

    Parameters
    ----------
    first_sid : int
    root_symbols : list[str]
    years : list[int]
    month_codes : dict[str -> int]

    Expiration dates are on the 20th of the month prior to the month code.
    Notice dates are are on the 20th two months prior to the month code.
    Start dates are one year before the contract month.

    See Also
    --------
    make_future_info
    """
    nineteen_days = pd.Timedelta(days=19)
    one_year = pd.Timedelta(days=365)
    return make_future_info(
        first_sid=first_sid,
        root_symbols=root_symbols,
        years=years,
        notice_date_func=lambda dt: dt - MonthBegin(2) + nineteen_days,
        expiration_date_func=lambda dt: dt - MonthBegin(1) + nineteen_days,
        start_date_func=lambda dt: dt - one_year,
        month_codes=month_codes,
    )
