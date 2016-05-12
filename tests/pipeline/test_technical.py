from abc import abstractmethod

import numpy as np
from operator import itemgetter
import pandas as pd
import talib
from toolz import compose, excepts

from zipline.lib.adjusted_array import AdjustedArray
from zipline.lib.adjustment import Float64Add
from zipline.pipeline import TermGraph
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.term import AssetExists
from zipline.pipeline.factors import BollingerBands
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

    @abstractmethod
    def test_without_adjustments(self):
        raise NotImplementedError('test_without_adjustments')

    @abstractmethod
    def test_with_adjustments(self):
        raise NotImplementedError('test_with_adjustments')


class BollingerBandsTestCase(WithTechnicalFactor, ZiplineTestCase):
    cases = parameter_space(
        window_length={5, 10, 20},
        k={1.5, 2, 2.5},
        mask_sid={True, False},
    )

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

    def _allnans(self, e):
        """Handler for toolz.excepts that returns three all nan arrays.
        """
        nans = np.full(self.ndays, np.nan)
        return nans, nans, nans

    def expected(self, window_length, k, closes):
        """Compute the expected data (without adjustments) for the given
        window, k, and closes array.

        This uses talib.BBANDS to generate the expected data.
        """
        return map(
            # Stack all of our uppers, middles, lowers into three 2d arrays
            # whose columns are the sids. After that, slice off only the
            # window care about.
            compose(
                itemgetter(np.s_[window_length - 1:]),
                np.column_stack,
            ),
            # Take our sequence of [(uppers, middles, lowers)] per sid and
            # turn it into three sequences of:
            # uppers for all sids, middles for all sids, lowers for all sids.
            zip(*(
                # talib breaks when the input array is all nan and raises
                # an instance of Exception. We catch that here and return
                # three all nan arrays instead.
                excepts(Exception, talib.BBANDS, self._allnans)(
                    closes[:, n],
                    window_length,
                    k,
                    k,
                )
                for n in range(self.nassets)
            )),
        )

    @cases
    def test_without_adjustments(self, window_length, k, mask_sid):
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

        assert_equal(
            result.upper,
            expected_upper,
        )
        assert_equal(
            result.middle,
            expected_middle,
        )
        assert_equal(
            result.lower,
            expected_lower,
        )

    @cases
    def test_with_adjustments(self, window_length, k, mask_sid):
        closes = self.closes(mask_sid)
        adjustment_offset = 5
        adjustment_idx = self.ndays - adjustment_offset
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
                    {
                        adjustment_idx: [
                            Float64Add(
                                first_row=0,
                                last_row=adjustment_idx,
                                first_col=0,
                                last_col=self.nassets - 1,
                                value=1000,
                            ),
                        ],
                    },
                    np.nan,
                ),
            },
            mask_sid=mask_sid,
        )['f']

        # Get the uppers, middles, and lowers without the adjustment applied.
        bases = self.expected(window_length, k, closes)
        adjusted_closes = closes.copy()
        adjusted_closes[:adjustment_idx + 1] += 1000
        # Get the uppers, middles, and lowers with the adjustment applied.
        adjusted = self.expected(window_length, k, adjusted_closes)

        # Create the actual expected data by using the unadjusted results up
        # to the adjument offset, then use the adjusted data for all indices
        # past the adjustment offset.
        expected_upper, expected_middle, expected_lower = (
            np.vstack((
                base[:-adjustment_offset],
                adjusted[-adjustment_offset:],
            ))
            for base, adjusted in zip(bases, adjusted)
        )

        assert_equal(
            result.upper,
            expected_upper,
        )
        assert_equal(
            result.middle,
            expected_middle,
        )
        assert_equal(
            result.lower,
            expected_lower,
        )
