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
from zipline.data.equities import USEquityPricing
from zipline.data.ffc.frame import DataFrameFFCLoader, MULTIPLY
from zipline.data.ffc.loaders.us_equity_pricing import (
    BcolzDailyBarReader,
    DailyBarWriterFromCSVs,
    SQLiteAdjustmentReader,
    SQLiteAdjustmentWriter,
    USEquityPricingLoader,
)
from zipline.finance import trading

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


class ClosesOnly(TestCase):

    def setUp(self):
        self.env = env = trading.TradingEnvironment()
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
        self.first_asset_start = min(asset_info.start_date)
        self.last_asset_end = max(asset_info.end_date)
        env.write_data(equities_df=asset_info)
        self.asset_finder = finder = env.asset_finder

        sids = (1, 2, 3)
        self.assets = finder.retrieve_all(sids)

        # View of the baseline data.
        self.closes = DataFrame(
            {sid: arange(1, len(self.dates) + 1) * sid for sid in sids},
            index=self.dates,
            dtype=float,
        )

        # Add a split for 'A' on its second date.
        self.split_asset = self.assets[0]
        self.split_date = self.split_asset.start_date + trading_day
        self.split_ratio = 0.5
        self.adjustments = DataFrame.from_records([
            {
                'sid': self.split_asset.sid,
                'value': self.split_ratio,
                'kind': MULTIPLY,
                'start_date': Timestamp('NaT'),
                'end_date': self.split_date,
                'apply_date': self.split_date,
            }
        ])

        # View of the data on/after the split.
        self.adj_closes = adj_closes = self.closes.copy()
        adj_closes.ix[:self.split_date, self.split_asset] *= self.split_ratio

        self.ffc_loader = DataFrameFFCLoader(
            column=USEquityPricing.close,
            baseline=self.closes,
            adjustments=self.adjustments,
        )

    def expected_close(self, date, asset):
        if date < self.split_date:
            lookup = self.closes
        else:
            lookup = self.adj_closes
        return lookup.loc[date, asset]

    def exists(self, date, asset):
        return asset.start_date <= date <= asset.end_date

    def test_assets_appear_on_correct_days(self):
        """
        Assert that assets appear at correct times during a backtest, with
        correctly-adjusted close price values.
        """
        def initialize(context):
            add_factor(USEquityPricing.close.latest, 'close')

        def handle_data(context, data):
            factors = data.factors
            date = get_datetime().normalize()
            for asset in self.assets:
                # Assets should appear iff they exist today and yesterday.
                exists_today = self.exists(date, asset)
                existed_yesterday = self.exists(date - trading_day, asset)
                if exists_today and existed_yesterday:
                    latest = factors.loc[asset, 'close']
                    self.assertEqual(latest, self.expected_close(date, asset))
                else:
                    self.assertNotIn(asset, factors.index)

        before_trading_start = handle_data

        algo = TradingAlgorithm(
            initialize=initialize,
            handle_data=handle_data,
            before_trading_start=before_trading_start,
            data_frequency='daily',
            ffc_loader=self.ffc_loader,
            start=self.first_asset_start - trading_day,
            end=self.last_asset_end + trading_day,
            env=self.env,
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
        cls.env = trading.TradingEnvironment()
        cls.env.write_data(equities_df=asset_info)
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
    def tearDownClass(cls):
        del cls.env
        cls.tempdir.cleanup()

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

    def make_source(self):
        return Panel(self.raw_data).tz_localize('UTC', axis=1)

    def test_handle_adjustment(self):
        AAPL, MSFT, BRK_A = assets = self.AAPL, self.MSFT, self.BRK_A

        # Our view of the data before AAPL's split on June 9, 2014.
        raw = {k: v.copy() for k, v in iteritems(self.raw_data)}

        split_date = Timestamp("2014-06-09", tz='UTC')
        split_loc = self.dates.get_loc(split_date)
        split_ratio = 7.0

        # Our view of the data after AAPL's split.  All prices from before June
        # 9 get divided by the split ratio, and volumes get multiplied by the
        # split ratio.
        adj = {k: v.copy() for k, v in iteritems(self.raw_data)}
        for column in 'open', 'high', 'low', 'close':
            adj[AAPL].ix[:split_loc, column] /= split_ratio
        adj[AAPL].ix[:split_loc, 'volume'] *= split_ratio

        window_lengths = [1, 2, 5, 10]
        # length -> asset -> expected vwap
        vwaps = {length: {} for length in window_lengths}
        vwap_keys = {}
        for length in window_lengths:
            vwap_keys[length] = "vwap_%d" % length
            for asset in AAPL, MSFT, BRK_A:
                raw_vwap = rolling_vwap(raw[asset], length)
                adj_vwap = rolling_vwap(adj[asset], length)
                # Shift computed results one day forward so that they're
                # labelled by the date on which they'll be seen in the
                # algorithm. (We can't show the close price for day N until day
                # N + 1.)
                vwaps[length][asset] = concat(
                    [
                        raw_vwap[:split_loc - 1],
                        adj_vwap[split_loc - 1:]
                    ]
                ).shift(1, trading_day)

        # Make sure all the expected vwaps have the same dates.
        vwap_dates = vwaps[1][self.AAPL].index
        for dict_ in itervalues(vwaps):
            # Each value is a dict mapping sid -> expected series.
            for series in itervalues(dict_):
                self.assertTrue((vwap_dates == series.index).all())

        # Spot check expectations near the AAPL split.
        # length 1 vwap for the morning before the split should be the close
        # price of the previous day.
        before_split = vwaps[1][AAPL].loc[split_date - trading_day]
        assert_almost_equal(before_split, 647.3499, decimal=2)
        assert_almost_equal(
            before_split,
            raw[AAPL].loc[split_date - (2 * trading_day), 'close'],
            decimal=2,
        )

        # length 1 vwap for the morning of the split should be the close price
        # of the previous day, **ADJUSTED FOR THE SPLIT**.
        on_split = vwaps[1][AAPL].loc[split_date]
        assert_almost_equal(on_split, 645.5700 / split_ratio, decimal=2)
        assert_almost_equal(
            on_split,
            raw[AAPL].loc[split_date - trading_day, 'close'] / split_ratio,
            decimal=2,
        )

        # length 1 vwap on the day after the split should be the as-traded
        # close on the split day.
        after_split = vwaps[1][AAPL].loc[split_date + trading_day]
        assert_almost_equal(after_split, 93.69999, decimal=2)
        assert_almost_equal(
            after_split,
            raw[AAPL].loc[split_date, 'close'],
            decimal=2,
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

        algo = TradingAlgorithm(
            initialize=initialize,
            handle_data=handle_data,
            before_trading_start=before_trading_start,
            data_frequency='daily',
            ffc_loader=self.ffc_loader,
            start=self.dates[max(window_lengths)],
            end=self.dates[-1],
        )

        algo.run(
            source=self.make_source(),
            # Yes, I really do want to use the start and end dates I passed to
            # TradingAlgorithm.
            overwrite_sim_params=False,
        )
