from __future__ import division

from nose_parameterized import parameterized
from six.moves import range
import numpy as np
import pandas as pd
import talib

from zipline.lib.adjusted_array import AdjustedArray
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import (
    BollingerBands,
    Aroon,
    FastStochasticOscillator,
    IchimokuKinkoHyo,
    LinearWeightedMovingAverage,
    RateOfChangePercentage,
    TrueRange,
)
from zipline.testing import parameter_space
from zipline.testing.fixtures import ZiplineTestCase
from zipline.testing.predicates import assert_equal

from .base import BasePipelineTestCase


class BollingerBandsTestCase(BasePipelineTestCase):

    def closes(self, mask_last_sid):
        data = self.arange_data(dtype=np.float64)
        if mask_last_sid:
            data[:, -1] = np.nan
        return data

    def expected_bbands(self, window_length, k, closes):
        """Compute the expected data (without adjustments) for the given
        window, k, and closes array.

        This uses talib.BBANDS to generate the expected data.
        """
        lower_cols = []
        middle_cols = []
        upper_cols = []

        ndates, nassets = closes.shape

        for n in range(nassets):
            close_col = closes[:, n]
            if np.isnan(close_col).all():
                # ta-lib doesn't deal well with all nans.
                upper, middle, lower = [np.full(ndates, np.nan)] * 3
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
        mask_last_sid={True, False},
        __fail_fast=True,
    )
    def test_bollinger_bands(self, window_length, k, mask_last_sid):
        closes = self.closes(mask_last_sid=mask_last_sid)
        mask = ~np.isnan(closes)
        bbands = BollingerBands(window_length=window_length, k=k)

        expected = self.expected_bbands(window_length, k, closes)

        self.check_terms(
            terms={
                'upper': bbands.upper,
                'middle': bbands.middle,
                'lower': bbands.lower,
            },
            expected={
                'upper': expected[0],
                'middle': expected[1],
                'lower': expected[2],
            },
            initial_workspace={
                USEquityPricing.close: AdjustedArray(
                    data=closes,
                    mask=mask,
                    adjustments={},
                    missing_value=np.nan,
                ),
            },
            mask=self.build_mask(mask),
        )

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


class TestFastStochasticOscillator(ZiplineTestCase):
    """
    Test the Fast Stochastic Oscillator
    """

    def test_fso_expected_basic(self):
        """
        Simple test of expected output from fast stochastic oscillator
        """
        fso = FastStochasticOscillator()

        today = pd.Timestamp('2015')
        assets = np.arange(3, dtype=np.float64)
        out = np.empty(shape=(3,), dtype=np.float64)

        highs = np.full((50, 3), 3, dtype=np.float64)
        lows = np.full((50, 3), 2, dtype=np.float64)
        closes = np.full((50, 3), 4, dtype=np.float64)

        fso.compute(today, assets, out, closes, lows, highs)

        # Expected %K
        assert_equal(out, np.full((3,), 200, dtype=np.float64))

    @parameter_space(seed=range(5))
    def test_fso_expected_with_talib(self, seed):
        """
        Test the output that is returned from the fast stochastic oscillator
        is the same as that from the ta-lib STOCHF function.
        """
        window_length = 14
        nassets = 6
        rng = np.random.RandomState(seed=seed)

        input_size = (window_length, nassets)

        # values from 9 to 12
        closes = 9.0 + (rng.random_sample(input_size) * 3.0)

        # Values from 13 to 15
        highs = 13.0 + (rng.random_sample(input_size) * 2.0)

        # Values from 6 to 8.
        lows = 6.0 + (rng.random_sample(input_size) * 2.0)

        expected_out_k = []
        for i in range(nassets):
            fastk, fastd = talib.STOCHF(
                high=highs[:, i],
                low=lows[:, i],
                close=closes[:, i],
                fastk_period=window_length,
                fastd_period=1,
            )

            expected_out_k.append(fastk[-1])
        expected_out_k = np.array(expected_out_k)

        today = pd.Timestamp('2015')
        out = np.empty(shape=(nassets,), dtype=np.float)
        assets = np.arange(nassets, dtype=np.float)

        fso = FastStochasticOscillator()
        fso.compute(
            today, assets, out, closes, lows, highs
        )

        assert_equal(out, expected_out_k, array_decimal=6)


class IchimokuKinkoHyoTestCase(ZiplineTestCase):
    def test_ichimoku_kinko_hyo(self):
        window_length = 52
        today = pd.Timestamp('2014', tz='utc')
        nassets = 5
        assets = pd.Index(np.arange(nassets))
        days_col = np.arange(window_length)[:, np.newaxis]
        highs = np.arange(nassets) + 2 + days_col
        closes = np.arange(nassets) + 1 + days_col
        lows = np.arange(nassets) + days_col

        tenkan_sen_length = 9
        kijun_sen_length = 26
        chikou_span_length = 26
        ichimoku_kinko_hyo = IchimokuKinkoHyo(
            window_length=window_length,
            tenkan_sen_length=tenkan_sen_length,
            kijun_sen_length=kijun_sen_length,
            chikou_span_length=chikou_span_length,
        )

        dtype = [
            ('tenkan_sen', 'f8'),
            ('kijun_sen', 'f8'),
            ('senkou_span_a', 'f8'),
            ('senkou_span_b', 'f8'),
            ('chikou_span', 'f8'),
        ]
        out = np.recarray(
            shape=(nassets,),
            dtype=dtype,
            buf=np.empty(shape=(nassets,), dtype=dtype),
        )
        ichimoku_kinko_hyo.compute(
            today,
            assets,
            out,
            highs,
            lows,
            closes,
            tenkan_sen_length,
            kijun_sen_length,
            chikou_span_length,
        )

        expected_tenkan_sen = np.array([
            (53 + 43) / 2,
            (54 + 44) / 2,
            (55 + 45) / 2,
            (56 + 46) / 2,
            (57 + 47) / 2,
        ])
        expected_kijun_sen = np.array([
            (53 + 26) / 2,
            (54 + 27) / 2,
            (55 + 28) / 2,
            (56 + 29) / 2,
            (57 + 30) / 2,
        ])
        expected_senkou_span_a = (expected_tenkan_sen + expected_kijun_sen) / 2
        expected_senkou_span_b = np.array([
            (53 + 0) / 2,
            (54 + 1) / 2,
            (55 + 2) / 2,
            (56 + 3) / 2,
            (57 + 4) / 2,
        ])
        expected_chikou_span = np.array([
            27.0,
            28.0,
            29.0,
            30.0,
            31.0,
        ])

        assert_equal(
            out.tenkan_sen,
            expected_tenkan_sen,
            msg='tenkan_sen',
        )
        assert_equal(
            out.kijun_sen,
            expected_kijun_sen,
            msg='kijun_sen',
        )
        assert_equal(
            out.senkou_span_a,
            expected_senkou_span_a,
            msg='senkou_span_a',
        )
        assert_equal(
            out.senkou_span_b,
            expected_senkou_span_b,
            msg='senkou_span_b',
        )
        assert_equal(
            out.chikou_span,
            expected_chikou_span,
            msg='chikou_span',
        )

    @parameter_space(
        arg={'tenkan_sen_length', 'kijun_sen_length', 'chikou_span_length'},
    )
    def test_input_validation(self, arg):
        window_length = 52

        with self.assertRaises(ValueError) as e:
            IchimokuKinkoHyo(**{arg: window_length + 1})

        assert_equal(
            str(e.exception),
            '%s must be <= the window_length: 53 > 52' % arg,
        )


class TestRateOfChangePercentage(ZiplineTestCase):
    @parameterized.expand([
        ('constant', [2.] * 10, 0.0),
        ('step', [2.] + [1.] * 9, -50.0),
        ('linear', [2. + x for x in range(10)], 450.0),
        ('quadratic', [2. + x**2 for x in range(10)], 4050.0),
    ])
    def test_rate_of_change_percentage(self, test_name, data, expected):
        window_length = len(data)

        rocp = RateOfChangePercentage(
            inputs=(USEquityPricing.close,),
            window_length=window_length,
        )
        today = pd.Timestamp('2014')
        assets = np.arange(5, dtype=np.int64)
        # broadcast data across assets
        data = np.array(data)[:, np.newaxis] * np.ones(len(assets))

        out = np.zeros(len(assets))
        rocp.compute(today, assets, out, data)
        assert_equal(out, np.full((len(assets),), expected))


class TestLinearWeightedMovingAverage(ZiplineTestCase):
    def test_wma1(self):
        wma1 = LinearWeightedMovingAverage(
            inputs=(USEquityPricing.close,),
            window_length=10
        )

        today = pd.Timestamp('2015')
        assets = np.arange(5, dtype=np.int64)

        data = np.ones((10, 5))
        out = np.zeros(data.shape[1])

        wma1.compute(today, assets, out, data)
        assert_equal(out, np.ones(5))

    def test_wma2(self):
        wma2 = LinearWeightedMovingAverage(
            inputs=(USEquityPricing.close,),
            window_length=10
        )

        today = pd.Timestamp('2015')
        assets = np.arange(5, dtype=np.int64)

        data = np.arange(50, dtype=np.float64).reshape((10, 5))
        out = np.zeros(data.shape[1])

        wma2.compute(today, assets, out, data)
        assert_equal(out, np.array([30.,  31.,  32.,  33.,  34.]))


class TestTrueRange(ZiplineTestCase):

    def test_tr_basic(self):
        tr = TrueRange()

        today = pd.Timestamp('2014')
        assets = np.arange(3, dtype=np.int64)
        out = np.empty(3, dtype=np.float64)

        highs = np.full((2, 3), 3.)
        lows = np.full((2, 3), 2.)
        closes = np.full((2, 3), 1.)

        tr.compute(today, assets, out, highs, lows, closes)
        assert_equal(out, np.full((3,), 2.))
