"""
Tests for Algorithms running the full FFC stack.
"""
from unittest import TestCase
from os.path import (
    dirname,
    join,
    realpath,
)

from numpy import (
    array,
    arange,
    full_like,
    nan,
)
from numpy.testing import assert_almost_equal
from pandas import (
    concat,
    DataFrame,
    date_range,
    DatetimeIndex,
    Panel,
    read_csv,
    Series,
    Timestamp,
)
from six import iteritems, itervalues
from testfixtures import TempDirectory

from zipline.algorithm import TradingAlgorithm
from zipline.api import (
    add_factor,
    get_datetime,
)
from zipline.assets import AssetFinder
from zipline.data.equities import USEquityPricing
from zipline.data.ffc.frame import DataFrameFFCLoader
from zipline.data.ffc.loaders.us_equity_pricing import (
    BcolzDailyBarReader,
    DailyBarWriterFromCSVs,
    SQLiteAdjustmentReader,
    SQLiteAdjustmentWriter,
    USEquityPricingLoader,
)
from zipline.modelling.factor.technical import VWAP
from zipline.utils.test_utils import (
    make_simple_asset_info,
    str_to_seconds,
)
from zipline.utils.tradingcalendar import (
    trading_day,
    trading_days,
)


TEST_RESOURCE_PATH = join(
    dirname(dirname(realpath(__file__))),  # zipline_repo/tests
    'resources',
    'modelling_inputs',
)


def rolling_vwap(df, length):
    "Simple rolling vwap implementation for testing"
    closes = df['close'].values
    volumes = df['volume'].values
    product = closes * volumes
    out = full_like(closes, nan)
    for upper_bound in range(length, len(closes) + 1):
        bounds = slice(upper_bound - length, upper_bound)
        out[upper_bound - 1] = product[bounds].sum() / volumes[bounds].sum()

    return Series(out, index=df.index)


class AssetLifetimesTestCase(TestCase):

    def setUp(self):
        self.dates = date_range(
            '2014-01-01', '2014-02-01', freq=trading_day, tz='UTC'
        )
        asset_info = DataFrame.from_records([
            {
                'sid': 1,
                'symbol': 'A',
                'asset_type': 'equity',
                'start_date': self.dates[10],
                'end_date': self.dates[13],
                'exchange': 'TEST',
            },
            {
                'sid': 2,
                'symbol': 'B',
                'asset_type': 'equity',
                'start_date': self.dates[11],
                'end_date': self.dates[14],
                'exchange': 'TEST',
            },
            {
                'sid': 3,
                'symbol': 'C',
                'asset_type': 'equity',
                'start_date': self.dates[12],
                'end_date': self.dates[15],
                'exchange': 'TEST',
            },
        ])
        self.asset_finder = finder = AssetFinder(asset_info)

        sids = (1, 2, 3)
        self.assets = finder.retrieve_all(sids)

        self.closes = DataFrame(
            {sid: arange(1, len(self.dates) + 1) * sid for sid in sids},
            index=self.dates,
            dtype=float,
        )
        self.ffc_loader = DataFrameFFCLoader(
            column=USEquityPricing.close,
            baseline=self.closes,
            adjustments=None,
        )

    def expected_close(self, date, asset):
        return self.closes.loc[date, asset]

    def exists(self, date, asset):
        return asset.start_date <= date <= asset.end_date

    def test_assets_appear_on_correct_days(self):
        """
        Assert that asset lifetimes match what shows up in a backtest.
        """
        def initialize(context):
            add_factor(USEquityPricing.close.latest, 'close')

        def handle_data(context, data):
            factors = data.factors
            date = get_datetime().normalize()
            prev_date = self.dates[self.dates.get_loc(date) - 1]
            for asset in self.assets:
                # Assets should appear iff they existed yesterday **and**
                # today.  Phrased another way: provide data for an asset if it
                # traded yesterday, and if yesterday was not the last trading
                # day for the asset.
                if self.exists(date, asset) and self.exists(prev_date, asset):
                    latest = factors.loc[asset, 'close']
                    # We should have data that's up to date as of yesterday.
                    self.assertEqual(
                        latest,
                        self.expected_close(prev_date, asset)
                    )
                else:
                    self.assertNotIn(asset, factors.index)

        before_trading_start = handle_data

        algo = TradingAlgorithm(
            initialize=initialize,
            handle_data=handle_data,
            before_trading_start=before_trading_start,
            data_frequency='daily',
            ffc_loader=self.ffc_loader,
            asset_finder=self.asset_finder,
            start=self.dates[10],
            end=self.dates[17],
        )

        # Run for a week in the middle of our data.
        algo.run(source=self.closes.iloc[10:17])


class FFCAlgorithmTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.AAPL = 1
        cls.MSFT = 2
        cls.BRK_A = 3
        cls.assets = [cls.AAPL, cls.MSFT, cls.BRK_A]
        asset_info = make_simple_asset_info(
            cls.assets,
            Timestamp('2014'),
            Timestamp('2015'),
            ['AAPL', 'MSFT', 'BRK_A'],
        )
        cls.asset_finder = AssetFinder(asset_info)
        cls.tempdir = tempdir = TempDirectory()
        tempdir.create()
        try:
            cls.raw_data, cls.bar_reader = cls.create_bar_reader(tempdir)
            cls.adj_reader = cls.create_adjustment_reader(tempdir)
            cls.ffc_loader = USEquityPricingLoader(
                cls.bar_reader, cls.adj_reader
            )
        except:
            cls.tempdir.cleanup()
            raise

        cls.dates = cls.raw_data[cls.AAPL].index.tz_localize('UTC')

    @classmethod
    def create_bar_reader(cls, tempdir):
        resources = {
            cls.AAPL: join(TEST_RESOURCE_PATH, 'AAPL.csv'),
            cls.MSFT: join(TEST_RESOURCE_PATH, 'MSFT.csv'),
            cls.BRK_A: join(TEST_RESOURCE_PATH, 'BRK-A.csv'),
        }
        raw_data = {
            asset: read_csv(path, parse_dates=['day']).set_index('day')
            for asset, path in iteritems(resources)
        }
        # Add 'price' column as an alias because all kinds of stuff in zipline
        # depends on it being present. :/
        for frame in raw_data.values():
            frame['price'] = frame['close']

        writer = DailyBarWriterFromCSVs(resources)
        data_path = tempdir.getpath('testdata.bcolz')
        table = writer.write(data_path, trading_days, cls.assets)
        return raw_data, BcolzDailyBarReader(table)

    @classmethod
    def create_adjustment_reader(cls, tempdir):
        dbpath = tempdir.getpath('adjustments.sqlite')
        writer = SQLiteAdjustmentWriter(dbpath)
        splits = DataFrame.from_records([
            {
                'effective_date': str_to_seconds('2014-06-09'),
                'ratio': (1 / 7.0),
                'sid': cls.AAPL,
            }
        ])
        mergers = dividends = DataFrame(
            {
                # Hackery to make the dtypes correct on an empty frame.
                'effective_date': array([], dtype=int),
                'ratio': array([], dtype=float),
                'sid': array([], dtype=int),
            },
            index=DatetimeIndex([], tz='UTC'),
            columns=['effective_date', 'ratio', 'sid'],
        )
        writer.write(splits, mergers, dividends)
        return SQLiteAdjustmentReader(dbpath)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def make_source(self):
        return Panel(self.raw_data).tz_localize('UTC', axis=1)

    def test_handle_adjustment(self):
        AAPL, MSFT, BRK_A = assets = self.AAPL, self.MSFT, self.BRK_A
        raw_data = self.raw_data
        adjusted_data = {k: v.copy() for k, v in iteritems(raw_data)}

        AAPL_split_date = Timestamp("2014-06-09", tz='UTC')
        split_loc = raw_data[AAPL].index.get_loc(AAPL_split_date)

        # Our view of AAPL's history changes after the split.
        ohlc = ['open', 'high', 'low', 'close']
        adjusted_data[AAPL].ix[:split_loc, ohlc] /= 7.0
        adjusted_data[AAPL].ix[:split_loc, ['volume']] *= 7.0

        window_lengths = [1, 2, 5, 10]
        # length -> asset -> expected vwap
        vwaps = {length: {} for length in window_lengths}
        vwap_keys = {}
        for length in window_lengths:
            vwap_keys[length] = "vwap_%d" % length
            for asset in AAPL, MSFT, BRK_A:
                raw = rolling_vwap(raw_data[asset], length)
                adj = rolling_vwap(adjusted_data[asset], length)
                vwaps[length][asset] = concat(
                    [
                        raw[:split_loc],
                        adj[split_loc:]
                    ]
                )

        vwap_dates = vwaps[1][self.AAPL].index
        # Make sure all the expected vwaps have the same dates.
        for dict_ in itervalues(vwaps):
            # Each value is a dict mapping sid -> expected series.
            for series in itervalues(dict_):
                self.assertTrue((vwap_dates == series.index).all())

        def initialize(context):
            context.vwaps = []
            for length, key in iteritems(vwap_keys):
                context.vwaps.append(VWAP(window_length=length))
                add_factor(context.vwaps[-1], name=key)

        def handle_data(context, data):
            today_loc = vwap_dates.get_loc(get_datetime())
            factors = data.factors
            for length, key in iteritems(vwap_keys):
                for asset in assets:
                    computed = factors.loc[asset, key]

                    # We should get the most recent values as of **YESTERDAY**.
                    expected = vwaps[length][asset].iloc[today_loc - 1]

                    # Only having two places of precision here is a bit
                    # unfortunate.
                    assert_almost_equal(computed, expected, decimal=2)

        # Do the same checks in before_trading_start
        before_trading_start = handle_data

        algo = TradingAlgorithm(
            initialize=initialize,
            handle_data=handle_data,
            before_trading_start=before_trading_start,
            data_frequency='daily',
            ffc_loader=self.ffc_loader,
            asset_finder=self.asset_finder,
            start=self.dates[max(window_lengths)],
            end=self.dates[-1],
        )

        algo.run(
            source=self.make_source(),
            # Yes, I really do want to use the start and end dates I passed to
            # TradingAlgorithm.
            overwrite_sim_params=False,
        )
