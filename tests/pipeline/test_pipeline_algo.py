"""
Tests for Algorithms using the Pipeline API.
"""
from os.path import (
    dirname,
    join,
    realpath,
)

from nose_parameterized import parameterized
import numpy as np
from numpy import (
    array,
    arange,
    full_like,
    float64,
    nan,
    uint32,
)
from numpy.testing import assert_almost_equal
import pandas as pd
from pandas import (
    concat,
    DataFrame,
    date_range,
    read_csv,
    Series,
    Timestamp,
)
from pandas.tseries.tools import normalize_date
from six import iteritems, itervalues

from zipline.api import (
    attach_pipeline,
    pipeline_output,
    get_datetime,
)
from zipline.errors import (
    AttachPipelineAfterInitialize,
    PipelineOutputDuringInitialize,
    NoSuchPipeline,
    DuplicatePipelineName,
)
from zipline.finance.trading import SimulationParameters
from zipline.lib.adjustment import MULTIPLY
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import VWAP
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.pipeline.loaders.equity_pricing_loader import (
    USEquityPricingLoader,
)
from zipline.testing import (
    str_to_seconds
)
from zipline.testing import (
    create_empty_splits_mergers_frame,
    FakeDataPortal,
)
from zipline.testing.fixtures import (
    WithMakeAlgo,
    WithAdjustmentReader,
    WithBcolzEquityDailyBarReaderFromCSVs,
    ZiplineTestCase,
)
from zipline.utils.calendars import get_calendar

TEST_RESOURCE_PATH = join(
    dirname(dirname(realpath(__file__))),  # zipline_repo/tests
    'resources',
    'pipeline_inputs',
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


class ClosesAndVolumes(WithMakeAlgo, ZiplineTestCase):
    sids = 1, 2, 3
    START_DATE = pd.Timestamp('2014-01-01', tz='utc')
    END_DATE = pd.Timestamp('2014-02-01', tz='utc')
    dates = date_range(START_DATE, END_DATE, freq=get_calendar("NYSE").day,
                       tz='utc')

    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    DATA_PORTAL_USE_MINUTE_DATA = False

    # FIXME: This currently uses benchmark returns from the trading
    # environment.
    BENCHMARK_SID = None

    @classmethod
    def make_equity_info(cls):
        cls.equity_info = ret = DataFrame.from_records([
            {
                'sid': 1,
                'symbol': 'A',
                'start_date': cls.dates[10],
                'end_date': cls.dates[13],
                'exchange': 'TEST',
            },
            {
                'sid': 2,
                'symbol': 'B',
                'start_date': cls.dates[11],
                'end_date': cls.dates[14],
                'exchange': 'TEST',
            },
            {
                'sid': 3,
                'symbol': 'C',
                'start_date': cls.dates[12],
                'end_date': cls.dates[15],
                'exchange': 'TEST',
            },
        ])
        return ret

    @classmethod
    def make_equity_daily_bar_data(cls):
        cls.closes = DataFrame(
            {sid: arange(1, len(cls.dates) + 1) * sid for sid in cls.sids},
            index=cls.dates,
            dtype=float,
        )
        cls.volumes = cls.closes * 1000
        for sid in cls.sids:
            yield sid, DataFrame(
                {
                    'open': cls.closes[sid].values,
                    'high': cls.closes[sid].values,
                    'low': cls.closes[sid].values,
                    'close': cls.closes[sid].values,
                    'volume': cls.volumes[sid].values,
                },
                index=cls.dates,
            )

    @classmethod
    def init_class_fixtures(cls):
        super(ClosesAndVolumes, cls).init_class_fixtures()
        cls.first_asset_start = min(cls.equity_info.start_date)
        cls.last_asset_end = max(cls.equity_info.end_date)
        cls.assets = cls.asset_finder.retrieve_all(cls.sids)

        cls.trading_day = cls.trading_calendar.day

        # Add a split for 'A' on its second date.
        cls.split_asset = cls.assets[0]
        cls.split_date = cls.split_asset.start_date + cls.trading_day
        cls.split_ratio = 0.5
        cls.adjustments = DataFrame.from_records([
            {
                'sid': cls.split_asset.sid,
                'value': cls.split_ratio,
                'kind': MULTIPLY,
                'start_date': Timestamp('NaT'),
                'end_date': cls.split_date,
                'apply_date': cls.split_date,
            }
        ])

        cls.default_sim_params = SimulationParameters(
            start_session=cls.first_asset_start,
            end_session=cls.last_asset_end,
            trading_calendar=cls.trading_calendar,
            emission_rate='daily',
            data_frequency='daily',
        )

    def make_algo_kwargs(self, **overrides):
        return self.merge_with_inherited_algo_kwargs(
            ClosesAndVolumes,
            suite_overrides=dict(
                sim_params=self.default_sim_params,
                get_pipeline_loader=lambda column: self.pipeline_close_loader,
            ),
            method_overrides=overrides,
        )

    def init_instance_fixtures(self):
        super(ClosesAndVolumes, self).init_instance_fixtures()

        # View of the data on/after the split.
        self.adj_closes = adj_closes = self.closes.copy()
        adj_closes.ix[:self.split_date, self.split_asset] *= self.split_ratio
        self.adj_volumes = adj_volumes = self.volumes.copy()
        adj_volumes.ix[:self.split_date, self.split_asset] *= self.split_ratio

        self.pipeline_close_loader = DataFrameLoader(
            column=USEquityPricing.close,
            baseline=self.closes,
            adjustments=self.adjustments,
        )

        self.pipeline_volume_loader = DataFrameLoader(
            column=USEquityPricing.volume,
            baseline=self.volumes,
            adjustments=self.adjustments,
        )

    def expected_close(self, date, asset):
        if date < self.split_date:
            lookup = self.closes
        else:
            lookup = self.adj_closes
        return lookup.loc[date, asset]

    def expected_volume(self, date, asset):
        if date < self.split_date:
            lookup = self.volumes
        else:
            lookup = self.adj_volumes
        return lookup.loc[date, asset]

    def exists(self, date, asset):
        return asset.start_date <= date <= asset.end_date

    def test_attach_pipeline_after_initialize(self):
        """
        Assert that calling attach_pipeline after initialize raises correctly.
        """
        def initialize(context):
            pass

        def late_attach(context, data):
            attach_pipeline(Pipeline(), 'test')
            raise AssertionError("Shouldn't make it past attach_pipeline!")

        algo = self.make_algo(
            initialize=initialize,
            handle_data=late_attach,
        )

        with self.assertRaises(AttachPipelineAfterInitialize):
            algo.run()

        def barf(context, data):
            raise AssertionError("Shouldn't make it past before_trading_start")

        algo = self.make_algo(
            initialize=initialize,
            before_trading_start=late_attach,
            handle_data=barf,
        )

        with self.assertRaises(AttachPipelineAfterInitialize):
            algo.run()

    def test_pipeline_output_after_initialize(self):
        """
        Assert that calling pipeline_output after initialize raises correctly.
        """
        def initialize(context):
            attach_pipeline(Pipeline(), 'test')
            pipeline_output('test')
            raise AssertionError("Shouldn't make it past pipeline_output()")

        def handle_data(context, data):
            raise AssertionError("Shouldn't make it past initialize!")

        def before_trading_start(context, data):
            raise AssertionError("Shouldn't make it past initialize!")

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            before_trading_start=before_trading_start,
        )

        with self.assertRaises(PipelineOutputDuringInitialize):
            algo.run()

    def test_get_output_nonexistent_pipeline(self):
        """
        Assert that calling add_pipeline after initialize raises appropriately.
        """
        def initialize(context):
            attach_pipeline(Pipeline(), 'test')

        def handle_data(context, data):
            raise AssertionError("Shouldn't make it past before_trading_start")

        def before_trading_start(context, data):
            pipeline_output('not_test')
            raise AssertionError("Shouldn't make it past pipeline_output!")

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            before_trading_start=before_trading_start,
        )

        with self.assertRaises(NoSuchPipeline):
            algo.run()

    @parameterized.expand([('default', None),
                           ('day', 1),
                           ('week', 5),
                           ('year', 252),
                           ('all_but_one_day', 'all_but_one_day'),
                           ('custom_iter', 'custom_iter')])
    def test_assets_appear_on_correct_days(self, test_name, chunks):
        """
        Assert that assets appear at correct times during a backtest, with
        correctly-adjusted close price values.
        """

        if chunks == 'all_but_one_day':
            chunks = (
                self.dates.get_loc(self.last_asset_end) -
                self.dates.get_loc(self.first_asset_start)
            ) - 1
        elif chunks == 'custom_iter':
            chunks = []
            st = np.random.RandomState(12345)
            remaining = (
                self.dates.get_loc(self.last_asset_end) -
                self.dates.get_loc(self.first_asset_start)
            )
            while remaining > 0:
                chunk = st.randint(3)
                chunks.append(chunk)
                remaining -= chunk

        def initialize(context):
            p = attach_pipeline(Pipeline(), 'test', chunks=chunks)
            p.add(USEquityPricing.close.latest, 'close')

        def handle_data(context, data):
            results = pipeline_output('test')
            date = get_datetime().normalize()
            for asset in self.assets:
                # Assets should appear iff they exist today and yesterday.
                exists_today = self.exists(date, asset)
                existed_yesterday = self.exists(date - self.trading_day, asset)
                if exists_today and existed_yesterday:
                    latest = results.loc[asset, 'close']
                    self.assertEqual(latest, self.expected_close(date, asset))
                else:
                    self.assertNotIn(asset, results.index)

        before_trading_start = handle_data

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            before_trading_start=before_trading_start,
        )

        # Run for a week in the middle of our data.
        algo.run()

    def test_multiple_pipelines(self):
        """
        Test that we can attach multiple pipelines and access the correct
        output based on the pipeline name.
        """
        def initialize(context):
            pipeline_close = attach_pipeline(Pipeline(), 'test_close')
            pipeline_volume = attach_pipeline(Pipeline(), 'test_volume')

            pipeline_close.add(USEquityPricing.close.latest, 'close')
            pipeline_volume.add(USEquityPricing.volume.latest, 'volume')

        def handle_data(context, data):
            closes = pipeline_output('test_close')
            volumes = pipeline_output('test_volume')
            date = get_datetime().normalize()
            for asset in self.assets:
                # Assets should appear iff they exist today and yesterday.
                exists_today = self.exists(date, asset)
                existed_yesterday = self.exists(date - self.trading_day, asset)
                if exists_today and existed_yesterday:
                    self.assertEqual(
                        closes.loc[asset, 'close'],
                        self.expected_close(date, asset)
                    )
                    self.assertEqual(
                        volumes.loc[asset, 'volume'],
                        self.expected_volume(date, asset)
                    )
                else:
                    self.assertNotIn(asset, closes.index)
                    self.assertNotIn(asset, volumes.index)

        column_to_loader = {
            USEquityPricing.close: self.pipeline_close_loader,
            USEquityPricing.volume: self.pipeline_volume_loader,
        }

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            get_pipeline_loader=lambda column: column_to_loader[column],
        )

        algo.run()

    def test_duplicate_pipeline_names(self):
        """
        Test that we raise an error when we try to attach a pipeline with a
        name that already exists for another attached pipeline.
        """
        def initialize(context):
            attach_pipeline(Pipeline(), 'test')
            attach_pipeline(Pipeline(), 'test')

        algo = self.make_algo(initialize=initialize)
        with self.assertRaises(DuplicatePipelineName):
            algo.run()


class MockDailyBarSpotReader(object):
    """
    A BcolzDailyBarReader which returns a constant value for spot price.
    """
    def get_value(self, sid, day, column):
        return 100.0


class PipelineAlgorithmTestCase(WithMakeAlgo,
                                WithBcolzEquityDailyBarReaderFromCSVs,
                                WithAdjustmentReader,
                                ZiplineTestCase):
    AAPL = 1
    MSFT = 2
    BRK_A = 3
    ASSET_FINDER_EQUITY_SIDS = AAPL, MSFT, BRK_A
    ASSET_FINDER_EQUITY_SYMBOLS = 'AAPL', 'MSFT', 'BRK_A'
    START_DATE = Timestamp('2014', tz='UTC')
    END_DATE = Timestamp('2015', tz='UTC')

    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    DATA_PORTAL_USE_MINUTE_DATA = False

    # FIXME: This currently uses benchmark returns from the trading
    # environment.
    BENCHMARK_SID = None

    @classmethod
    def make_equity_daily_bar_data(cls):
        resources = {
            cls.AAPL: join(TEST_RESOURCE_PATH, 'AAPL.csv'),
            cls.MSFT: join(TEST_RESOURCE_PATH, 'MSFT.csv'),
            cls.BRK_A: join(TEST_RESOURCE_PATH, 'BRK-A.csv'),
        }
        cls.raw_data = raw_data = {
            asset: read_csv(path, parse_dates=['day']).set_index('day')
            for asset, path in resources.items()
        }
        # Add 'price' column as an alias because all kinds of stuff in zipline
        # depends on it being present. :/
        for frame in raw_data.values():
            frame['price'] = frame['close']

        return resources

    @classmethod
    def make_splits_data(cls):
        return DataFrame.from_records([
            {
                'effective_date': str_to_seconds('2014-06-09'),
                'ratio': (1 / 7.0),
                'sid': cls.AAPL,
            }
        ])

    @classmethod
    def make_mergers_data(cls):
        return create_empty_splits_mergers_frame()

    @classmethod
    def make_dividends_data(cls):
        return pd.DataFrame(array([], dtype=[
            ('sid', uint32),
            ('amount', float64),
            ('record_date', 'datetime64[ns]'),
            ('ex_date', 'datetime64[ns]'),
            ('declared_date', 'datetime64[ns]'),
            ('pay_date', 'datetime64[ns]'),
        ]))

    @classmethod
    def init_class_fixtures(cls):
        super(PipelineAlgorithmTestCase, cls).init_class_fixtures()
        cls.pipeline_loader = USEquityPricingLoader(
            cls.bcolz_equity_daily_bar_reader,
            cls.adjustment_reader,
        )
        cls.dates = cls.raw_data[cls.AAPL].index.tz_localize('UTC')
        cls.AAPL_split_date = Timestamp("2014-06-09", tz='UTC')
        cls.assets = cls.asset_finder.retrieve_all(
            cls.ASSET_FINDER_EQUITY_SIDS
        )

    def make_algo_kwargs(self, **overrides):
        return self.merge_with_inherited_algo_kwargs(
            PipelineAlgorithmTestCase,
            suite_overrides=dict(
                get_pipeline_loader=lambda column: self.pipeline_loader,
            ),
            method_overrides=overrides,
        )

    def compute_expected_vwaps(self, window_lengths):
        AAPL, MSFT, BRK_A = self.AAPL, self.MSFT, self.BRK_A

        # Our view of the data before AAPL's split on June 9, 2014.
        raw = {k: v.copy() for k, v in iteritems(self.raw_data)}

        split_date = self.AAPL_split_date
        split_loc = self.dates.get_loc(split_date)
        split_ratio = 7.0

        # Our view of the data after AAPL's split.  All prices from before June
        # 9 get divided by the split ratio, and volumes get multiplied by the
        # split ratio.
        adj = {k: v.copy() for k, v in iteritems(self.raw_data)}
        for column in 'open', 'high', 'low', 'close':
            adj[AAPL].ix[:split_loc, column] /= split_ratio
        adj[AAPL].ix[:split_loc, 'volume'] *= split_ratio

        # length -> asset -> expected vwap
        vwaps = {length: {} for length in window_lengths}
        for length in window_lengths:
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
                ).shift(1, self.trading_calendar.day)

        # Make sure all the expected vwaps have the same dates.
        vwap_dates = vwaps[1][self.AAPL].index
        for dict_ in itervalues(vwaps):
            # Each value is a dict mapping sid -> expected series.
            for series in itervalues(dict_):
                self.assertTrue((vwap_dates == series.index).all())

        # Spot check expectations near the AAPL split.
        # length 1 vwap for the morning before the split should be the close
        # price of the previous day.
        before_split = vwaps[1][AAPL].loc[split_date -
                                          self.trading_calendar.day]
        assert_almost_equal(before_split, 647.3499, decimal=2)
        assert_almost_equal(
            before_split,
            raw[AAPL].loc[split_date - (2 * self.trading_calendar.day),
                          'close'],
            decimal=2,
        )

        # length 1 vwap for the morning of the split should be the close price
        # of the previous day, **ADJUSTED FOR THE SPLIT**.
        on_split = vwaps[1][AAPL].loc[split_date]
        assert_almost_equal(on_split, 645.5700 / split_ratio, decimal=2)
        assert_almost_equal(
            on_split,
            raw[AAPL].loc[split_date -
                          self.trading_calendar.day, 'close'] / split_ratio,
            decimal=2,
        )

        # length 1 vwap on the day after the split should be the as-traded
        # close on the split day.
        after_split = vwaps[1][AAPL].loc[split_date +
                                         self.trading_calendar.day]
        assert_almost_equal(after_split, 93.69999, decimal=2)
        assert_almost_equal(
            after_split,
            raw[AAPL].loc[split_date, 'close'],
            decimal=2,
        )

        return vwaps

    @parameterized.expand([
        (True,),
        (False,),
    ])
    def test_handle_adjustment(self, set_screen):
        AAPL, MSFT, BRK_A = assets = self.assets

        window_lengths = [1, 2, 5, 10]
        vwaps = self.compute_expected_vwaps(window_lengths)

        def vwap_key(length):
            return "vwap_%d" % length

        def initialize(context):
            pipeline = Pipeline()
            context.vwaps = []
            for length in vwaps:
                name = vwap_key(length)
                factor = VWAP(window_length=length)
                context.vwaps.append(factor)
                pipeline.add(factor, name=name)

            filter_ = (USEquityPricing.close.latest > 300)
            pipeline.add(filter_, 'filter')
            if set_screen:
                pipeline.set_screen(filter_)

            attach_pipeline(pipeline, 'test')

        def handle_data(context, data):
            today = normalize_date(get_datetime())
            results = pipeline_output('test')
            expect_over_300 = {
                AAPL: today < self.AAPL_split_date,
                MSFT: False,
                BRK_A: True,
            }
            for asset in assets:
                should_pass_filter = expect_over_300[asset]
                if set_screen and not should_pass_filter:
                    self.assertNotIn(asset, results.index)
                    continue

                asset_results = results.loc[asset]
                self.assertEqual(asset_results['filter'], should_pass_filter)
                for length in vwaps:
                    computed = results.loc[asset, vwap_key(length)]
                    expected = vwaps[length][asset].loc[today]
                    # Only having two places of precision here is a bit
                    # unfortunate.
                    assert_almost_equal(computed, expected, decimal=2)

        # Do the same checks in before_trading_start
        before_trading_start = handle_data

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            before_trading_start=before_trading_start,
            sim_params=SimulationParameters(
                start_session=self.dates[max(window_lengths)],
                end_session=self.dates[-1],
                data_frequency='daily',
                emission_rate='daily',
                trading_calendar=self.trading_calendar,
            )
        )

        algo.run(
            FakeDataPortal(self.env),
            # Yes, I really do want to use the start and end dates I passed to
            # self.make_algo.
            overwrite_sim_params=False,
        )

    def test_empty_pipeline(self):

        # For ensuring we call before_trading_start.
        count = [0]

        def initialize(context):
            pipeline = attach_pipeline(Pipeline(), 'test')

            vwap = VWAP(window_length=10)
            pipeline.add(vwap, 'vwap')

            # Nothing should have prices less than 0.
            pipeline.set_screen(vwap < 0)

        def handle_data(context, data):
            pass

        def before_trading_start(context, data):
            context.results = pipeline_output('test')
            self.assertTrue(context.results.empty)
            count[0] += 1

        self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
            before_trading_start=before_trading_start,
            sim_params=SimulationParameters(
                start_session=self.dates[0],
                end_session=self.dates[-1],
                data_frequency='daily',
                emission_rate='daily',
                trading_calendar=self.trading_calendar,
            )
        )

        self.assertTrue(count[0] > 0)

    def test_pipeline_beyond_daily_bars(self):
        """
        Ensure that we can run an algo with pipeline beyond the max date
        of the daily bars.
        """

        # For ensuring we call before_trading_start.
        count = [0]

        current_day = self.trading_calendar.next_session_label(
            self.pipeline_loader.raw_price_loader.last_available_dt,
        )

        def initialize(context):
            pipeline = attach_pipeline(Pipeline(), 'test')

            vwap = VWAP(window_length=10)
            pipeline.add(vwap, 'vwap')

            # Nothing should have prices less than 0.
            pipeline.set_screen(vwap < 0)

        def handle_data(context, data):
            pass

        def before_trading_start(context, data):
            context.results = pipeline_output('test')
            self.assertTrue(context.results.empty)
            count[0] += 1

        self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
            before_trading_start=before_trading_start,
            sim_params=SimulationParameters(
                start_session=self.dates[0],
                end_session=current_day,
                data_frequency='daily',
                emission_rate='daily',
                trading_calendar=self.trading_calendar,
            )
        )

        self.assertTrue(count[0] > 0)
