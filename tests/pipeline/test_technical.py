import numpy as np
import pandas as pd
import talib

from zipline.lib.adjusted_array import AdjustedArray
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
            try:
                upper, middle, lower = talib.BBANDS(
                    closes[:, n],
                    window_length,
                    k,
                    k,
                )
            except Exception:
                # If the input array is all nan then talib raises an instance
                # of Exception.
                upper, middle, lower = [np.full(self.ndays, np.nan)] * 3

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
