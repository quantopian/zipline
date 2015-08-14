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
    full_like,
    nan,
)
from numpy.testing import assert_almost_equal
from pandas import (
    concat,
    DataFrame,
    DatetimeIndex,
    Panel,
    read_csv,
    Series,
    Timestamp,
)
from six import iteritems
from testfixtures import TempDirectory

from zipline.algorithm import TradingAlgorithm
from zipline.api import (
    #    add_filter,
    add_factor,
    get_datetime,
)
from zipline.assets import AssetFinder
# from zipline.data.equities import USEquityPricing
from zipline.data.ffc.loaders.us_equity_pricing import (
    BcolzDailyBarReader,
    DailyBarWriterFromCSVs,
    SQLiteAdjustmentReader,
    SQLiteAdjustmentWriter,
    USEquityPricingLoader,
)
# from zipline.modelling.factor import CustomFactor
from zipline.finance import trading
from zipline.modelling.factor.technical import VWAP
from zipline.utils.test_utils import (
    make_simple_asset_info,
    str_to_seconds,
)
from zipline.utils.tradingcalendar import trading_days


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
        cls.env = trading.TradingEnvironment()
        cls.env.write_data(equities_df=asset_info)
        cls.asset_finder = AssetFinder(cls.env.engine)
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

        def initialize(context):
            context.vwaps = []
            for length, key in iteritems(vwap_keys):
                context.vwaps.append(VWAP(window_length=length))
                add_factor(context.vwaps[-1], name=key)

        def handle_data(context, data):
            today = get_datetime()
            factors = data.factors
            for length, key in iteritems(vwap_keys):
                for asset in assets:
                    computed = factors.loc[asset, key]
                    expected = vwaps[length][asset].loc[today]
                    # Only having two places of precision here is a bit
                    # unfortunate.
                    assert_almost_equal(computed, expected, decimal=2)

        # Do the same checks in before_trading_start
        before_trading_start = handle_data

        # Create fresh trading environment as the algo.run()
        # method will attempt to write data to disk, and could
        # violate SQL constraints.
        trading.environment = trading.TradingEnvironment()

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
