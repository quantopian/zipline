"""
Tests for zipline.pipeline.loaders.frame.DataFrameLoader.
"""
from unittest import mock
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from zipline.utils.calendar_utils import get_calendar

from zipline.lib.adjustment import (
    ADD,
    Float64Add,
    Float64Multiply,
    Float64Overwrite,
    MULTIPLY,
    OVERWRITE,
)
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.domain import US_EQUITIES
from zipline.pipeline.loaders.frame import DataFrameLoader

import pytest


@pytest.fixture(scope="class")
def frame_loader(request):
    request.cls.trading_day = get_calendar("NYSE").day
    request.cls.nsids = 5
    request.cls.ndates = 20
    request.cls.sids = pd.Index(range(request.cls.nsids), dtype="int64")
    request.cls.dates = pd.date_range(
        start="2014-01-02",
        freq=request.cls.trading_day,
        periods=request.cls.ndates,
    )
    request.cls.mask = np.ones(
        (len(request.cls.dates), len(request.cls.sids)), dtype=bool
    )


@pytest.mark.usefixtures("frame_loader")
class TestDataFrameLoader:
    def test_bad_input(self):
        data = np.arange(100).reshape(self.ndates, self.nsids)
        baseline = pd.DataFrame(data, index=self.dates, columns=self.sids)
        loader = DataFrameLoader(
            USEquityPricing.close,
            baseline,
        )

        with pytest.raises(ValueError):
            # Wrong column.
            loader.load_adjusted_array(
                US_EQUITIES,
                [USEquityPricing.open],
                self.dates,
                self.sids,
                self.mask,
            )

        with pytest.raises(ValueError):
            # Too many columns.
            loader.load_adjusted_array(
                US_EQUITIES,
                [USEquityPricing.open, USEquityPricing.close],
                self.dates,
                self.sids,
                self.mask,
            )

    def test_baseline(self):
        data = np.arange(100).reshape(self.ndates, self.nsids)
        baseline = pd.DataFrame(data, index=self.dates, columns=self.sids)
        loader = DataFrameLoader(USEquityPricing.close, baseline)

        dates_slice = slice(None, 10, None)
        sids_slice = slice(1, 3, None)
        [adj_array] = loader.load_adjusted_array(
            US_EQUITIES,
            [USEquityPricing.close],
            self.dates[dates_slice],
            self.sids[sids_slice],
            self.mask[dates_slice, sids_slice],
        ).values()

        for idx, window in enumerate(adj_array.traverse(window_length=3)):
            expected = baseline.values[dates_slice, sids_slice][idx : idx + 3]
            assert_array_equal(window, expected)

    def test_adjustments(self):
        data = np.arange(100).reshape(self.ndates, self.nsids)
        baseline = pd.DataFrame(data, index=self.dates, columns=self.sids)

        # Use the dates from index 10 on and sids 1-3.
        dates_slice = slice(10, None, None)
        sids_slice = slice(1, 4, None)

        # Adjustments that should actually affect the output.
        relevant_adjustments = [
            {
                "sid": 1,
                "start_date": None,
                "end_date": self.dates[15],
                "apply_date": self.dates[16],
                "value": 0.5,
                "kind": MULTIPLY,
            },
            {
                "sid": 2,
                "start_date": self.dates[5],
                "end_date": self.dates[15],
                "apply_date": self.dates[16],
                "value": 1.0,
                "kind": ADD,
            },
            {
                "sid": 2,
                "start_date": self.dates[15],
                "end_date": self.dates[16],
                "apply_date": self.dates[17],
                "value": 1.0,
                "kind": ADD,
            },
            {
                "sid": 3,
                "start_date": self.dates[16],
                "end_date": self.dates[17],
                "apply_date": self.dates[18],
                "value": 99.0,
                "kind": OVERWRITE,
            },
        ]

        # These adjustments shouldn't affect the output.
        irrelevant_adjustments = [
            {  # Sid Not Requested
                "sid": 0,
                "start_date": self.dates[16],
                "end_date": self.dates[17],
                "apply_date": self.dates[18],
                "value": -9999.0,
                "kind": OVERWRITE,
            },
            {  # Sid Unknown
                "sid": 9999,
                "start_date": self.dates[16],
                "end_date": self.dates[17],
                "apply_date": self.dates[18],
                "value": -9999.0,
                "kind": OVERWRITE,
            },
            {  # Date Not Requested
                "sid": 2,
                "start_date": self.dates[1],
                "end_date": self.dates[2],
                "apply_date": self.dates[3],
                "value": -9999.0,
                "kind": OVERWRITE,
            },
            {  # Date Before Known Data
                "sid": 2,
                "start_date": self.dates[0] - (2 * self.trading_day),
                "end_date": self.dates[0] - self.trading_day,
                "apply_date": self.dates[0] - self.trading_day,
                "value": -9999.0,
                "kind": OVERWRITE,
            },
            {  # Date After Known Data
                "sid": 2,
                "start_date": self.dates[-1] + self.trading_day,
                "end_date": self.dates[-1] + (2 * self.trading_day),
                "apply_date": self.dates[-1] + (3 * self.trading_day),
                "value": -9999.0,
                "kind": OVERWRITE,
            },
        ]

        adjustments = pd.DataFrame(relevant_adjustments + irrelevant_adjustments)
        loader = DataFrameLoader(
            USEquityPricing.close,
            baseline,
            adjustments=adjustments,
        )

        expected_baseline = baseline.iloc[dates_slice, sids_slice]

        formatted_adjustments = loader.format_adjustments(
            self.dates[dates_slice],
            self.sids[sids_slice],
        )
        expected_formatted_adjustments = {
            6: [
                Float64Multiply(
                    first_row=0,
                    last_row=5,
                    first_col=0,
                    last_col=0,
                    value=0.5,
                ),
                Float64Add(
                    first_row=0,
                    last_row=5,
                    first_col=1,
                    last_col=1,
                    value=1.0,
                ),
            ],
            7: [
                Float64Add(
                    first_row=5,
                    last_row=6,
                    first_col=1,
                    last_col=1,
                    value=1.0,
                ),
            ],
            8: [
                Float64Overwrite(
                    first_row=6,
                    last_row=7,
                    first_col=2,
                    last_col=2,
                    value=99.0,
                )
            ],
        }
        assert formatted_adjustments == expected_formatted_adjustments

        mask = self.mask[dates_slice, sids_slice]
        with mock.patch("zipline.pipeline.loaders.frame.AdjustedArray") as m:
            loader.load_adjusted_array(
                US_EQUITIES,
                columns=[USEquityPricing.close],
                dates=self.dates[dates_slice],
                sids=self.sids[sids_slice],
                mask=mask,
            )

        assert m.call_count == 1

        args, kwargs = m.call_args
        assert_array_equal(kwargs["data"], expected_baseline.values)
        assert kwargs["adjustments"] == expected_formatted_adjustments
