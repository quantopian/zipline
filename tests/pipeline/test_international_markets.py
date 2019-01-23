"""Tests for pipelines on international markets.
"""
import numpy as np
import pandas as pd

from zipline.assets.synthetic import (
    make_rotating_equity_info,
    make_multi_exchange_equity_info,
)
from zipline.data.in_memory_daily_bars import InMemoryDailyBarReader
from zipline.pipeline.domain import (
    CA_EQUITIES,
    GB_EQUITIES,
    US_EQUITIES,
)
from zipline.pipeline import Pipeline
from zipline.pipeline.data import EquityPricing, USEquityPricing
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders.equity_pricing_loader import EquityPricingLoader
from zipline.pipeline.loaders.synthetic import NullAdjustmentReader
from zipline.testing.predicates import assert_equal
from zipline.testing.core import (
    parameter_space,
    random_tick_prices,
)

import zipline.testing.fixtures as zf


def T(s):
    return pd.Timestamp(s, tz='UTC')


class WithInternationalDailyBarData(zf.WithAssetFinder):
    """
    Fixture for generating international daily bars.

    Eventually this should be moved into zipline.testing.fixtures and should
    replace most of the existing machinery
    """
    DAILY_BAR_START_DATE = zf.alias('START_DATE')
    DAILY_BAR_END_DATE = zf.alias('END_DATE')
    DAILY_BAR_LOOKBACK_DAYS = 0

    INTERNATIONAL_PRICING_STARTING_PRICES = {
        'XNYS': 100,  # NYSE
        'XTSE': 50,   # Toronto Stock Exchange
        'XLON': 25,   # London Stock Exchange
    }

    @classmethod
    def make_daily_bar_data(cls, assets, calendar, sessions):
        # Generate prices corresponding to uniform random returns with a slight
        # positive tendency.
        start = cls.INTERNATIONAL_PRICING_STARTING_PRICES[calendar.name]

        closes = random_tick_prices(start, len(sessions))
        opens = closes - 0.05
        highs = closes + 0.10
        lows = closes - 0.10
        volumes = np.arange(10000, 10000 + len(closes))

        base_frame = pd.DataFrame({
            'close': closes,
            'open': opens,
            'high': highs,
            'low': lows,
            'volume': volumes,
        }, index=sessions)

        for asset in assets:
            sid = asset.sid
            yield sid, base_frame + sid

    @classmethod
    def init_class_fixtures(cls):
        super(WithInternationalDailyBarData, cls).init_class_fixtures()

        cls.daily_bar_sessions = {}
        cls.daily_bar_data = {}
        cls.daily_bar_readers = {}

        for calendar, assets, in cls.assets_by_calendar.items():
            name = calendar.name
            start_delta = cls.DAILY_BAR_LOOKBACK_DAYS * calendar.day
            start_session = cls.DAILY_BAR_START_DATE - start_delta

            sessions = calendar.sessions_in_range(
                start_session, cls.DAILY_BAR_END_DATE,
            )

            cls.daily_bar_sessions[name] = sessions
            cls.daily_bar_data[name] = dict(cls.make_daily_bar_data(
                assets=assets, calendar=calendar, sessions=sessions,
            ))

            panel = (pd.Panel.from_dict(cls.daily_bar_data[name])
                     .transpose(2, 1, 0))
            cls.daily_bar_readers[name] = InMemoryDailyBarReader.from_panel(
                panel,
                calendar,
            )


class WithInternationalPricingPipelineEngine(WithInternationalDailyBarData):

    @classmethod
    def init_class_fixtures(cls):
        (super(WithInternationalPricingPipelineEngine, cls)
         .init_class_fixtures())

        adjustments = NullAdjustmentReader()
        cls.loaders = {
            GB_EQUITIES: EquityPricingLoader(
                cls.daily_bar_readers['XLON'],
                adjustments,
            ),
            US_EQUITIES: EquityPricingLoader(
                cls.daily_bar_readers['XNYS'],
                adjustments,
            ),
            CA_EQUITIES: EquityPricingLoader(
                cls.daily_bar_readers['XTSE'],
                adjustments,
            )
        }
        cls.engine = SimplePipelineEngine(
            get_loader=cls.get_loader,
            asset_finder=cls.asset_finder,
        )

    @classmethod
    def get_loader(cls, column):
        return cls.loaders[column.domain]

    def run_pipeline(self, pipeline, start_date, end_date):
        return self.engine.run_pipeline(pipeline, start_date, end_date)


class InternationalEquityTestCase(WithInternationalPricingPipelineEngine,
                                  zf.ZiplineTestCase):
    START_DATE = T('2014-01-02')
    END_DATE = T('2014-02-06')  # Chosen to match the asset setup data below.

    EXCHANGE_INFO = pd.DataFrame.from_records([
        {'exchange': 'XNYS', 'country_code': 'US'},
        {'exchange': 'XTSE', 'country_code': 'CA'},
        {'exchange': 'XLON', 'country_code': 'GB'},
    ])

    @classmethod
    def make_equity_info(cls):
        # - 20 assets on each exchange.
        # - Each asset lives for 5 days.
        # - A new asset starts each day.
        out = make_multi_exchange_equity_info(
            factory=make_rotating_equity_info,
            exchange_sids={
                'XNYS': range(20),
                'XTSE': range(20, 40),
                'XLON': range(40, 60),
            },
            first_start=cls.START_DATE,
            periods_between_starts=1,
            # NOTE: The asset_lifetime parameter name is a bit misleading. It
            #       determines the number of trading days between each asset's
            #       start_date and end_date, so assets created with this method
            #       actual "live" for (asset_lifetime + 1) days. But, since
            #       pipeline doesn't show you an asset the day it IPOs, this
            #       number matches the number of days that each asset should
            #       appear in a pipeline output.
            asset_lifetime=5,
        )
        assert_equal(out.end_date.max(), cls.END_DATE)
        return out

    @classmethod
    def make_exchanges_info(cls, equities, futures, root_symbols):
        return cls.EXCHANGE_INFO

    @parameter_space(domain=[CA_EQUITIES, US_EQUITIES, GB_EQUITIES])
    def test_generic_pipeline_with_explicit_domain(self, domain):
        calendar = domain.calendar
        pipe = Pipeline({
            'open': EquityPricing.open.latest,
            'high': EquityPricing.high.latest,
            'low': EquityPricing.low.latest,
            'close': EquityPricing.close.latest,
            'volume': EquityPricing.volume.latest,
        }, domain=domain)

        sessions = self.daily_bar_sessions[calendar.name]

        # Run the pipeline for a 7 day chunk in the middle of our data.
        #
        # Using this region ensures that there are assets that never appear in
        # the pipeline both because they end too soon, and because they start
        # too late.
        start, end = sessions[[-17, -10]]
        result = self.run_pipeline(pipe, start, end)

        all_assets = self.assets_by_calendar[calendar]

        # We expect the index of the result to contain all assets that were
        # alive during the interval between our start and end (not including
        # the asset's IPO date).
        expected_assets = [
            a for a in all_assets
            if alive_in_range(a, start, end, include_asset_start_date=False)
        ]
        # off by 1 from above to be inclusive of the end date
        expected_dates = sessions[-17:-9]

        for col in pipe.columns:
            # result_data should look like this:
            #
            #     E     F     G     H     I     J     K     L     M     N     O     P # noqa
            # 24.17 25.17 26.17 27.17 28.17   NaN   NaN   NaN   NaN   NaN   NaN   NaN # noqa
            #   NaN 25.18 26.18 27.18 28.18 29.18   NaN   NaN   NaN   NaN   NaN   NaN # noqa
            #   NaN   NaN 26.23 27.23 28.23 29.23 30.23   NaN   NaN   NaN   NaN   NaN # noqa
            #   NaN   NaN   NaN 27.28 28.28 29.28 30.28 31.28   NaN   NaN   NaN   NaN # noqa
            #   NaN   NaN   NaN   NaN 28.30 29.30 30.30 31.30 32.30   NaN   NaN   NaN # noqa
            #   NaN   NaN   NaN   NaN   NaN 29.29 30.29 31.29 32.29 33.29   NaN   NaN # noqa
            #   NaN   NaN   NaN   NaN   NaN   NaN 30.27 31.27 32.27 33.27 34.27   NaN # noqa
            #   NaN   NaN   NaN   NaN   NaN   NaN   NaN 31.29 32.29 33.29 34.29 35.29 # noqa
            result_data = result[col].unstack()

            # Check indices.
            assert_equal(pd.Index(expected_assets), result_data.columns)
            assert_equal(expected_dates, result_data.index)

            # Check values.
            for asset in expected_assets:
                for date in expected_dates:
                    value = result_data.at[date, asset]
                    self.check_expected_latest_value(
                        calendar, col, date, asset, value,
                    )

    def test_explicit_specialization_matches_implicit(self):
        pipeline_specialized = Pipeline({
            'open': EquityPricing.open.latest,
            'high': EquityPricing.high.latest,
            'low': EquityPricing.low.latest,
            'close': EquityPricing.close.latest,
            'volume': EquityPricing.volume.latest,
        }, domain=US_EQUITIES)
        dataset_specialized = Pipeline({
            'open': USEquityPricing.open.latest,
            'high': USEquityPricing.high.latest,
            'low': USEquityPricing.low.latest,
            'close': USEquityPricing.close.latest,
            'volume': USEquityPricing.volume.latest,
        })

        sessions = self.daily_bar_sessions['XNYS']
        self.assert_identical_results(
            pipeline_specialized,
            dataset_specialized,
            sessions[1],
            sessions[-1],
        )

    def check_expected_latest_value(self, calendar, col, date, asset, value):
        """Check the expected result of column.latest from a pipeline.
        """
        if np.isnan(value):
            # If we got a NaN, we should be outside the asset's
            # lifetime.
            self.assertTrue(date <= asset.start_date or date > asset.end_date)
        else:
            self.assertTrue(asset.start_date < date <= asset.end_date)
            bars = self.daily_bar_data[calendar.name]
            # Subtract a day because pipeline shows values as of the morning
            expected_value = bars[asset.sid].loc[date - calendar.day, col]
            assert_equal(value, expected_value)

    def assert_identical_results(self, left, right, start_date, end_date):
        """Assert that two pipelines produce the same results.
        """
        left_result = self.run_pipeline(left, start_date, end_date)
        right_result = self.run_pipeline(right, start_date, end_date)
        assert_equal(left_result, right_result)


def alive_in_range(asset, start, end, include_asset_start_date=False):
    """
    Check if an asset was alive in the range from start to end.

    Parameters
    ----------
    asset : Asset
        The asset to check
    start : pd.Timestamp
        Start of the interval.
    end : pd.Timestamp
        End of the interval.
    include_asset_start_date : bool
        Whether to include the start date of the asset when checking liveness.

    Returns
    -------
    was_alive : bool
        Whether or not ``asset`` was alive for any days in the range from
        ``start`` to ``end``.
    """
    if include_asset_start_date:
        asset_start = asset.start_date
    else:
        asset_start = asset.start_date + pd.Timedelta('1 day')
    return intervals_overlap((asset_start, asset.end_date), (start, end))


def intervals_overlap(a, b):
    """
    Check whether a pair of datetime intervals overlap.

    Parameters
    ----------
    a : (pd.Timestamp, pd.Timestamp)
    b : (pd.Timestamp, pd.Timestamp)

    Returns
    -------
    have_overlap : bool
        Bool indicating whether there there is a non-empty intersection between
        the intervals.
    """
    # If the intervals do not overlap, then either the first is strictly before
    # the second, or the second is strictly before the first.
    a_strictly_before = a[1] < b[0]
    b_strictly_before = b[1] < a[0]
    return not (a_strictly_before or b_strictly_before)
