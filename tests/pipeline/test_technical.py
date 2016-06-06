from __future__ import division

from nose_parameterized import parameterized
import numpy as np
import pandas as pd
import talib

from zipline.lib.adjusted_array import AdjustedArray
from zipline.pipeline import TermGraph
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.term import AssetExists
from zipline.pipeline.factors import (
    BollingerBands,
    Aroon,
    FastStochasticOscillator
)
from zipline.testing import ExplodingObject, parameter_space
from zipline.testing.fixtures import WithAssetFinder, ZiplineTestCase
from zipline.testing.predicates import assert_equal


class WithTechnicalFactor(WithAssetFinder):
    """ZiplineTestCase fixture for testing technical factors.
    """
    ASSET_FINDER_EQUITY_SIDS = tuple(range(5))
    START_DATE = pd.Timestamp('2014-01-01', tz='utc')

    @classmethod
    def init_class_fixtures(cls):
        super(WithTechnicalFactor, cls).init_class_fixtures()
        cls.ndays = ndays = 24
        cls.nassets = nassets = len(cls.ASSET_FINDER_EQUITY_SIDS)
        cls.dates = dates = pd.date_range(cls.START_DATE, periods=ndays)
        cls.assets = pd.Index(cls.asset_finder.sids)
        cls.engine = SimplePipelineEngine(
            lambda column: ExplodingObject(),
            dates,
            cls.asset_finder,
        )
        cls.asset_exists = exists = np.full((ndays, nassets), True, dtype=bool)
        cls.asset_exists_masked = masked = exists.copy()
        masked[:, -1] = False

    def run_graph(self, graph, initial_workspace, mask_sid):
        initial_workspace.setdefault(
            AssetExists(),
            self.asset_exists_masked if mask_sid else self.asset_exists,
        )
        return self.engine.compute_chunk(
            graph,
            self.dates,
            self.assets,
            initial_workspace,
        )


class BollingerBandsTestCase(WithTechnicalFactor, ZiplineTestCase):
    @classmethod
    def init_class_fixtures(cls):
        super(BollingerBandsTestCase, cls).init_class_fixtures()
        cls._closes = closes = (
            np.arange(cls.ndays, dtype=float)[:, np.newaxis] +
            np.arange(cls.nassets, dtype=float) * 100
        )
        cls._closes_masked = masked = closes.copy()
        masked[:, -1] = np.nan

    def closes(self, masked):
        return self._closes_masked if masked else self._closes

    def expected(self, window_length, k, closes):
        """Compute the expected data (without adjustments) for the given
        window, k, and closes array.

        This uses talib.BBANDS to generate the expected data.
        """
        lower_cols = []
        middle_cols = []
        upper_cols = []
        for n in range(self.nassets):
            close_col = closes[:, n]
            if np.isnan(close_col).all():
                # ta-lib doesn't deal well with all nans.
                upper, middle, lower = [np.full(self.ndays, np.nan)] * 3
            else:
                upper, middle, lower = talib.BBANDS(
                    close_col,
                    window_length,
                    k,
                    k,
                )

            upper_cols.append(upper)
            middle_cols.append(middle)
            lower_cols.append(lower)

        # Stack all of our uppers, middles, lowers into three 2d arrays
        # whose columns are the sids. After that, slice off only the
        # rows we care about.
        where = np.s_[window_length - 1:]
        uppers = np.column_stack(upper_cols)[where]
        middles = np.column_stack(middle_cols)[where]
        lowers = np.column_stack(lower_cols)[where]
        return uppers, middles, lowers

    @parameter_space(
        window_length={5, 10, 20},
        k={1.5, 2, 2.5},
        mask_sid={True, False},
    )
    def test_bollinger_bands(self, window_length, k, mask_sid):
        closes = self.closes(mask_sid)
        result = self.run_graph(
            TermGraph({
                'f': BollingerBands(
                    window_length=window_length,
                    k=k,
                ),
            }),
            initial_workspace={
                USEquityPricing.close: AdjustedArray(
                    closes,
                    np.full_like(closes, True, dtype=bool),
                    {},
                    np.nan,
                ),
            },
            mask_sid=mask_sid,
        )['f']

        expected_upper, expected_middle, expected_lower = self.expected(
            window_length,
            k,
            closes,
        )

        assert_equal(result.upper, expected_upper)
        assert_equal(result.middle, expected_middle)
        assert_equal(result.lower, expected_lower)

    def test_bollinger_bands_output_ordering(self):
        bbands = BollingerBands(window_length=5, k=2)
        lower, middle, upper = bbands
        self.assertIs(lower, bbands.lower)
        self.assertIs(middle, bbands.middle)
        self.assertIs(upper, bbands.upper)


class AroonTestCase(ZiplineTestCase):
    window_length = 10
    nassets = 5
    dtype = [('down', 'f8'), ('up', 'f8')]

    @parameterized.expand([
        (np.arange(window_length),
         np.arange(window_length) + 1,
         np.recarray(shape=(nassets,), dtype=dtype,
                     buf=np.array([0, 100] * nassets, dtype='f8'))),
        (np.arange(window_length, 0, -1),
         np.arange(window_length, 0, -1) - 1,
         np.recarray(shape=(nassets,), dtype=dtype,
                     buf=np.array([100, 0] * nassets, dtype='f8'))),
        (np.array([10, 10, 10, 1, 10, 10, 10, 10, 10, 10]),
         np.array([1, 1, 1, 1, 1, 10, 1, 1, 1, 1]),
         np.recarray(shape=(nassets,), dtype=dtype,
                     buf=np.array([100 * 3 / 9, 100 * 5 / 9] * nassets,
                                  dtype='f8'))),
    ])
    def test_aroon_basic(self, lows, highs, expected_out):
        aroon = Aroon(window_length=self.window_length)
        today = pd.Timestamp('2014', tz='utc')
        assets = pd.Index(np.arange(self.nassets, dtype=np.int64))
        shape = (self.nassets,)
        out = np.recarray(shape=shape, dtype=self.dtype,
                          buf=np.empty(shape=shape, dtype=self.dtype))

        aroon.compute(today, assets, out, lows, highs)

        assert_equal(out, expected_out)


class TestFastStochasticOscillator(WithTechnicalFactor, ZiplineTestCase):
    """
    Test the Fast Stochastic Oscillator
    """

    def test_fso_expected_basic(self):
        """
        Simple test of expected output from fast stochastic oscillator
        """
        fso = FastStochasticOscillator()

        today = pd.Timestamp('2015')
        assets = np.arange(3, dtype=np.float)
        out = np.empty(shape=(3,), dtype=np.float)

        highs = np.full((50, 3), 3)
        lows = np.full((50, 3), 2)
        closes = np.full((50, 3), 4)

        fso.compute(today, assets, out, closes, lows, highs)

        # Expected %K
        assert_equal(out, np.full((3,), 200))

    def test_fso_expected_with_talib(self):
        """
        Test the output that is returned from the fast stochastic oscillator
        is the same as that from the ta-lib STOCHF function.
        """
        window_length = 14
        nassets = 6
        closes = np.random.random_integers(1, 6, size=(50, nassets))*1.0
        highs = np.random.random_integers(4, 6, size=(50, nassets))*1.0
        lows = np.random.random_integers(1, 3, size=(50, nassets))*1.0

        expected_out_k = []
        for i in range(nassets):
            e = talib.STOCHF(
                high=highs[:, i],
                low=lows[:, i],
                close=closes[:, i],
                fastk_period=window_length,
            )

            expected_out_k.append(e[0][-1])
        expected_out_k = np.array(expected_out_k)

        today = pd.Timestamp('2015')
        out = np.empty(shape=(nassets,), dtype=np.float)
        assets = np.arange(nassets, dtype=np.float)

        fso = FastStochasticOscillator()
        fso.compute(
            today, assets, out, closes, lows, highs
        )

        assert_equal(out, expected_out_k)
