"""
PipelineLoader accepting a DataFrame as input.
"""
from functools import partial

from numpy import (
    ix_,
    zeros,
)
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Int64Index,
)
from zipline.lib.adjusted_array import AdjustedArray
from zipline.lib.adjustment import make_adjustment_from_labels
from zipline.utils.pandas_utils import sort_values
from .base import PipelineLoader

ADJUSTMENT_COLUMNS = Index([
    'sid',
    'value',
    'kind',
    'start_date',
    'end_date',
    'apply_date',
])


class DataFrameLoader(PipelineLoader):
    """
    A PipelineLoader that reads its input from DataFrames.

    Mostly useful for testing, but can also be used for real work if your data
    fits in memory.

    Parameters
    ----------
    column : zipline.pipeline.data.BoundColumn
        The column whose data is loadable by this loader.
    baseline : pandas.DataFrame
        A DataFrame with index of type DatetimeIndex and columns of type
        Int64Index.  Dates should be labelled with the first date on which a
        value would be **available** to an algorithm.  This means that OHLCV
        data should generally be shifted back by a trading day before being
        supplied to this class.

    adjustments : pandas.DataFrame, default=None
        A DataFrame with the following columns:
            sid : int
            value : any
            kind : int (zipline.pipeline.loaders.frame.ADJUSTMENT_TYPES)
            start_date : datetime64 (can be NaT)
            end_date : datetime64 (must be set)
            apply_date : datetime64 (must be set)

        The default of None is interpreted as "no adjustments to the baseline".
    """

    def __init__(self, column, baseline, adjustments=None):
        self.column = column
        self.baseline = baseline.values.astype(self.column.dtype)
        self.dates = baseline.index
        self.assets = baseline.columns

        if adjustments is None:
            adjustments = DataFrame(
                index=DatetimeIndex([]),
                columns=ADJUSTMENT_COLUMNS,
            )
        else:
            # Ensure that columns are in the correct order.
            adjustments = adjustments.reindex_axis(ADJUSTMENT_COLUMNS, axis=1)
            sort_values(adjustments, ['apply_date', 'sid'], inplace=True)

        self.adjustments = adjustments
        self.adjustment_apply_dates = DatetimeIndex(adjustments.apply_date)
        self.adjustment_end_dates = DatetimeIndex(adjustments.end_date)
        self.adjustment_sids = Int64Index(adjustments.sid)

    def format_adjustments(self, dates, assets):
        """
        Build a dict of Adjustment objects in the format expected by
        AdjustedArray.

        Returns a dict of the form:
        {
            # Integer index into `dates` for the date on which we should
            # apply the list of adjustments.
            1 : [
                Float64Multiply(first_row=2, last_row=4, col=3, value=0.5),
                Float64Overwrite(first_row=3, last_row=5, col=1, value=2.0),
                ...
            ],
            ...
        }
        """
        make_adjustment = partial(make_adjustment_from_labels, dates, assets)

        min_date, max_date = dates[[0, -1]]
        # TODO: Consider porting this to Cython.
        if len(self.adjustments) == 0:
            return {}

        # Mask for adjustments whose apply_dates are in the requested window of
        # dates.
        date_bounds = self.adjustment_apply_dates.slice_indexer(
            min_date,
            max_date,
        )
        dates_filter = zeros(len(self.adjustments), dtype='bool')
        dates_filter[date_bounds] = True
        # Ignore adjustments whose apply_date is in range, but whose end_date
        # is out of range.
        dates_filter &= (self.adjustment_end_dates >= min_date)

        # Mask for adjustments whose sids are in the requested assets.
        sids_filter = self.adjustment_sids.isin(assets.values)

        adjustments_to_use = self.adjustments.loc[
            dates_filter & sids_filter
        ].set_index('apply_date')

        # For each apply_date on which we have an adjustment, compute
        # the integer index of that adjustment's apply_date in `dates`.
        # Then build a list of Adjustment objects for that apply_date.
        # This logic relies on the sorting applied on the previous line.
        out = {}
        previous_apply_date = object()
        for row in adjustments_to_use.itertuples():
            # This expansion depends on the ordering of the DataFrame columns,
            # defined above.
            apply_date, sid, value, kind, start_date, end_date = row
            if apply_date != previous_apply_date:
                # Get the next apply date if no exact match.
                row_loc = dates.get_loc(apply_date, method='bfill')
                current_date_adjustments = out[row_loc] = []
                previous_apply_date = apply_date

            # Look up the approprate Adjustment constructor based on the value
            # of `kind`.
            current_date_adjustments.append(
                make_adjustment(start_date, end_date, sid, kind, value)
            )
        return out

    def load_adjusted_array(self, columns, dates, assets, mask):
        """
        Load data from our stored baseline.
        """
        column = self.column
        if len(columns) != 1:
            raise ValueError(
                "Can't load multiple columns with DataFrameLoader"
            )
        elif columns[0] != column:
            raise ValueError("Can't load unknown column %s" % columns[0])

        date_indexer = self.dates.get_indexer(dates)
        assets_indexer = self.assets.get_indexer(assets)

        # Boolean arrays with True on matched entries
        good_dates = (date_indexer != -1)
        good_assets = (assets_indexer != -1)

        return {
            column: AdjustedArray(
                # Pull out requested columns/rows from our baseline data.
                data=self.baseline[ix_(date_indexer, assets_indexer)],
                # Mask out requested columns/rows that didnt match.
                mask=(good_assets & good_dates[:, None]) & mask,
                adjustments=self.format_adjustments(dates, assets),
                missing_value=column.missing_value,
            ),
        }
