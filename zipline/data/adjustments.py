# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from errno import ENOENT
from os import remove
from os.path import exists

import sqlite3

import pandas as pd
from numpy import (
    floating,
    integer,
    issubdtype,
    uint32
)
import logbook
import numpy as np

from six import iteritems

from zipline.data.us_equity_pricing import NoDataOnDate

logger = logbook.Logger('Adjustments')


SQLITE_ADJUSTMENT_COLUMNS = frozenset(['effective_date', 'ratio', 'sid'])
SQLITE_ADJUSTMENT_COLUMN_DTYPES = {
    'effective_date': integer,
    'ratio': floating,
    'sid': integer,
}
SQLITE_ADJUSTMENT_TABLENAMES = frozenset(['splits', 'dividends', 'mergers'])


SQLITE_DIVIDEND_PAYOUT_COLUMNS = frozenset(
    ['sid',
     'ex_date',
     'declared_date',
     'pay_date',
     'record_date',
     'amount'])
SQLITE_DIVIDEND_PAYOUT_COLUMN_DTYPES = {
    'sid': integer,
    'ex_date': integer,
    'declared_date': integer,
    'record_date': integer,
    'pay_date': integer,
    'amount': float,
}


SQLITE_STOCK_DIVIDEND_PAYOUT_COLUMNS = frozenset(
    ['sid',
     'ex_date',
     'declared_date',
     'record_date',
     'pay_date',
     'payment_sid',
     'ratio'])
SQLITE_STOCK_DIVIDEND_PAYOUT_COLUMN_DTYPES = {
    'sid': integer,
    'ex_date': integer,
    'declared_date': integer,
    'record_date': integer,
    'pay_date': integer,
    'payment_sid': integer,
    'ratio': float,
}


class SQLiteAdjustmentWriter(object):
    """
    Writer for data to be read by SQLiteAdjustmentWriter

    Parameters
    ----------
    conn_or_path : str or sqlite3.Connection
        A handle to the target sqlite database.
    overwrite : bool, optional, default=False
        If True and conn_or_path is a string, remove any existing files at the
        given path before connecting.

    See Also
    --------
    SQLiteAdjustmentReader
    """

    def __init__(self, conn_or_path, trading_days,
                 daily_bar_spot_reader, overwrite=False):
        if isinstance(conn_or_path, sqlite3.Connection):
            self.conn = conn_or_path
        elif isinstance(conn_or_path, str):
            if overwrite and exists(conn_or_path):
                try:
                    remove(conn_or_path)
                except OSError as e:
                    if e.errno != ENOENT:
                        raise
            self.conn = sqlite3.connect(conn_or_path)
        else:
            raise TypeError("Unknown connection type %s" % type(conn_or_path))

        self.trading_days = trading_days
        self.daily_bar_spot_reader = daily_bar_spot_reader

    def write_frame(self, tablename, frame):
        if frozenset(frame.columns) != SQLITE_ADJUSTMENT_COLUMNS:
            raise ValueError(
                "Unexpected frame columns:\n"
                "Expected Columns: %s\n"
                "Received Columns: %s" % (
                    SQLITE_ADJUSTMENT_COLUMNS,
                    frame.columns.tolist(),
                )
            )
        elif tablename not in SQLITE_ADJUSTMENT_TABLENAMES:
            raise ValueError(
                "Adjustment table %s not in %s" % (
                    tablename, SQLITE_ADJUSTMENT_TABLENAMES
                )
            )

        expected_dtypes = SQLITE_ADJUSTMENT_COLUMN_DTYPES
        actual_dtypes = frame.dtypes
        for colname, expected in iteritems(expected_dtypes):
            actual = actual_dtypes[colname]
            if not issubdtype(actual, expected):
                raise TypeError(
                    "Expected data of type {expected} for column '{colname}', "
                    "but got {actual}.".format(
                        expected=expected,
                        colname=colname,
                        actual=actual,
                    )
                )
        return frame.to_sql(tablename, self.conn)

    def write_dividend_payouts(self, frame):
        """
        Write dividend payout data to SQLite table `dividend_payouts`.
        """
        if frozenset(frame.columns) != SQLITE_DIVIDEND_PAYOUT_COLUMNS:
            raise ValueError(
                "Unexpected frame columns:\n"
                "Expected Columns: %s\n"
                "Received Columns: %s" % (
                    sorted(SQLITE_DIVIDEND_PAYOUT_COLUMNS),
                    sorted(frame.columns.tolist()),
                )
            )

        expected_dtypes = SQLITE_DIVIDEND_PAYOUT_COLUMN_DTYPES
        actual_dtypes = frame.dtypes
        for colname, expected in iteritems(expected_dtypes):
            actual = actual_dtypes[colname]
            if not issubdtype(actual, expected):
                raise TypeError(
                    "Expected data of type {expected} for column '{colname}', "
                    "but got {actual}.".format(
                        expected=expected,
                        colname=colname,
                        actual=actual,
                    )
                )
        return frame.to_sql('dividend_payouts', self.conn)

    def write_stock_dividend_payouts(self, frame):
        if frozenset(frame.columns) != SQLITE_STOCK_DIVIDEND_PAYOUT_COLUMNS:
            raise ValueError(
                "Unexpected frame columns:\n"
                "Expected Columns: %s\n"
                "Received Columns: %s" % (
                    sorted(SQLITE_STOCK_DIVIDEND_PAYOUT_COLUMNS),
                    sorted(frame.columns.tolist()),
                )
            )

        expected_dtypes = SQLITE_STOCK_DIVIDEND_PAYOUT_COLUMN_DTYPES
        actual_dtypes = frame.dtypes
        for colname, expected in iteritems(expected_dtypes):
            actual = actual_dtypes[colname]
            if not issubdtype(actual, expected):
                raise TypeError(
                    "Expected data of type {expected} for column '{colname}', "
                    "but got {actual}.".format(
                        expected=expected,
                        colname=colname,
                        actual=actual,
                    )
                )
        return frame.to_sql('stock_dividend_payouts', self.conn)

    def calc_dividend_ratios(self, dividends):
        """
        Calculate the ratios to apply to equities when looking back at pricing
        history so that the price is smoothed over the ex_date, when the market
        adjusts to the change in equity value due to upcoming dividend.

        Returns
        -------
        DataFrame
            A frame in the same format as splits and mergers, with keys
            - sid, the id of the equity
            - effective_date, the date in seconds on which to apply the ratio.
            - ratio, the ratio to apply to backwards looking pricing data.
        """
        ex_dates = dividends.ex_date.values

        sids = dividends.sid.values
        amounts = dividends.amount.values

        ratios = np.full(len(amounts), np.nan)

        spot_reader = self.daily_bar_spot_reader

        trading_days = self.trading_days

        for i, amount in enumerate(amounts):
            sid = sids[i]
            ex_date = ex_dates[i]
            day_loc = trading_days.get_loc(ex_date)
            div_adj_date = trading_days[day_loc - 1]
            try:
                prev_close = spot_reader.unadjusted_spot_price(
                    sid, div_adj_date, 'close')
                ratio = 1.0 - amount / (prev_close)
                ratios[i] = ratio
            except NoDataOnDate:
                logger.warn("Couldn't compute ratio for dividend %s" % {
                    'sid': sid,
                    'ex_date': ex_date,
                    'gross_amount': amount,
                })
                continue

        effective_dates = ex_dates.astype('datetime64[s]').astype(uint32)

        return pd.DataFrame({
            'sid': sids,
            'effective_date': effective_dates,
            'ratio': ratios,
        })

    def write_dividend_data(self, dividends, stock_dividends=None):
        """
        Write both dividend payouts and the derived price adjustment ratios.
        """

        # First write the dividend payouts.
        dividend_payouts = dividends.copy()
        dividend_payouts['ex_date'] = dividend_payouts['ex_date'].values.\
            astype('datetime64[s]').astype(integer)
        dividend_payouts['record_date'] = \
            dividend_payouts['record_date'].values.astype('datetime64[s]').\
            astype(integer)
        dividend_payouts['declared_date'] = \
            dividend_payouts['declared_date'].values.astype('datetime64[s]').\
            astype(integer)
        dividend_payouts['pay_date'] = \
            dividend_payouts['declared_date'].values.astype('datetime64[s]').\
            astype(integer)

        self.write_dividend_payouts(dividend_payouts)

        if stock_dividends is not None:
            stock_dividend_payouts = stock_dividends.copy()
            stock_dividend_payouts['ex_date'] = \
                stock_dividend_payouts['ex_date'].values.\
                astype('datetime64[s]').astype(integer)
            stock_dividend_payouts['record_date'] = \
                stock_dividend_payouts['record_date'].values.\
                astype('datetime64[s]').astype(integer)
            stock_dividend_payouts['declared_date'] = \
                stock_dividend_payouts['declared_date'].\
                values.astype('datetime64[s]').astype(integer)
            stock_dividend_payouts['pay_date'] = \
                stock_dividend_payouts['declared_date'].\
                values.astype('datetime64[s]').astype(integer)
        else:
            stock_dividend_payouts = pd.DataFrame({
                'sid': np.array([], dtype=uint32),
                'record_date': np.array([], dtype=uint32),
                'ex_date': np.array([], dtype=uint32),
                'declared_date': np.array([], dtype=uint32),
                'pay_date': np.array([], dtype=uint32),
                'payment_sid': np.array([], dtype=uint32),
                'ratio': np.array([], dtype=float),
            })

        self.write_stock_dividend_payouts(stock_dividend_payouts)

        # Second from the dividend payouts, calculate ratios.

        dividend_ratios = self.calc_dividend_ratios(dividends)

        self.write_frame('dividends', dividend_ratios)

    def write(self, splits, mergers, dividends, stock_dividends=None):
        """
        Writes data to a SQLite file to be read by SQLiteAdjustmentReader.

        Parameters
        ----------
        splits : pandas.DataFrame
            Dataframe containing split data.
        mergers : pandas.DataFrame
            DataFrame containing merger data.
        dividends : pandas.DataFrame
            DataFrame containing dividend data.

        Notes
        -----
        DataFrame input (`splits`, `mergers`) should all have the following
        columns:

        effective_date : int
            The date, represented as seconds since Unix epoch, on which the
            adjustment should be applied.
        ratio : float
            A value to apply to all data earlier than the effective date.
        sid : int
            The asset id associated with this adjustment.

        The ratio column is interpreted as follows:
        - For all adjustment types, multiply price fields ('open', 'high',
          'low', and 'close') by the ratio.
        - For **splits only**, **divide** volume by the adjustment ratio.


        DataFrame input, 'dividends' should have the following columns:

        sid : int
            The asset id associated with this adjustment.
        ex_date : datetime64
            The date on which an equity must be held to be eligible to receive
            payment.
        declared_date : datetime64
            The date on which the dividend is announced to the public.
        pay_date : datetime64
            The date on which the dividend is distributed.
        record_date : datetime64
            The date on which the stock ownership is checked to determine
            distribution of dividends.
        amount : float
            The cash amount paid for each share.

        Dividend ratios should be calculated as
        1.0 - (dividend_value / "close on day prior to dividend ex_date").


        DataFrame input, 'stock_dividends' should have the following columns:

        sid : int
            The asset id associated with this adjustment.
        ex_date : datetime64
            The date on which an equity must be held to be eligible to receive
            payment.
        declared_date : datetime64
            The date on which the dividend is announced to the public.
        pay_date : datetime64
            The date on which the dividend is distributed.
        record_date : datetime64
            The date on which the stock ownership is checked to determine
            distribution of dividends.
        payment_sid : int
            The asset id of the shares that should be paid instead of cash.
        ratio: float
            The ration of currently held shares in the held sid that should
            be paid with new shares of the payment_sid.

        stock_dividends is optional.

        Returns
        -------
        None

        See Also
        --------
        SQLiteAdjustmentReader : Consumer for the data written by this class
        """
        self.write_frame('splits', splits)
        self.write_frame('mergers', mergers)
        self.write_dividend_data(dividends, stock_dividends)
        self.conn.execute(
            "CREATE INDEX splits_sids "
            "ON splits(sid)"
        )
        self.conn.execute(
            "CREATE INDEX splits_effective_date "
            "ON splits(effective_date)"
        )
        self.conn.execute(
            "CREATE INDEX mergers_sids "
            "ON mergers(sid)"
        )
        self.conn.execute(
            "CREATE INDEX mergers_effective_date "
            "ON mergers(effective_date)"
        )
        self.conn.execute(
            "CREATE INDEX dividends_sid "
            "ON dividends(sid)"
        )
        self.conn.execute(
            "CREATE INDEX dividends_effective_date "
            "ON dividends(effective_date)"
        )
        self.conn.execute(
            "CREATE INDEX dividend_payouts_sid "
            "ON dividend_payouts(sid)"
        )
        self.conn.execute(
            "CREATE INDEX dividend_payouts_ex_date "
            "ON dividend_payouts(ex_date)"
        )
        self.conn.execute(
            "CREATE INDEX dividend_payouts_record_date "
            "ON dividend_payouts(record_date)"
        )
        self.conn.execute(
            "CREATE INDEX stock_dividend_payouts_sid "
            "ON stock_dividend_payouts(sid)"
        )
        self.conn.execute(
            "CREATE INDEX stock_dividend_payouts_ex_date "
            "ON stock_dividend_payouts(ex_date)"
        )
        self.conn.execute(
            "CREATE INDEX stock_dividend_payouts_record_date "
            "ON stock_dividend_payouts(record_date)"
        )

    def close(self):
        self.conn.close()
