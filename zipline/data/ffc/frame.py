"""
FFC Loader accepting a DataFrame as input.
"""
from numpy import (
    ix_,
    uint8,
    zeros,
    zeros_like,
)
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Int64Index,
)
from zipline.data.adjusted_array import adjusted_array
from zipline.data.adjustment import (
    Float64Add,
    Float64Multiply,
    Float64Overwrite,
)
from zipline.data.ffc.base import FFCLoader


ADD, MULTIPLY, OVERWRITE = range(3)
ADJUSTMENT_CONSTRUCTORS = {
    ADD: Float64Add.from_assets_and_dates,
    MULTIPLY: Float64Multiply.from_assets_and_dates,
    OVERWRITE: Float64Overwrite.from_assets_and_dates,
}
ADJUSTMENT_COLUMNS = Index([
    'sid',
    'value',
    'kind',
    'start_date',
    'end_date',
    'apply_date',
])


class DataFrameFFCLoader(FFCLoader):
    """
    An FFCLoader that reads its input from DataFrames.

    Parameters
    ----------
    column : zipline.data.dataset.BoundColumn
        The column whose data is loadable by this loader.

    baseline : pandas.DataFrame
        A DataFrame with index of type DatetimeIndex and columns of type
        Int64Index.

    mask : numpy.ndarray[uint8], default=None
        A mask array of the same shape as baseline containing 0s for invalid
        values and 1 for valid values.

        The default of None is interpreted as "all inputs valid".

    adjustments : pandas.DataFrame, default=None
        A DataFrame with the following columns:
            sid : int
            value : any
            kind : int (zipline.data.ffc.frame.ADJUSTMENT_TYPES)
            start_date : datetime64 (can be NaT)
            end_date : datetime64 (must be set)
            apply_date : datetime64 (must be set)

        The default of None is interpreted as "no adjustments to the baseline".
    """

    def __init__(self, column, baseline, mask=None, adjustments=None):
        self.column = column
        self.baseline = baseline.values
        self.dates = baseline.index
        self.assets = baseline.columns

        if mask is not None:
            assert mask.dtype == uint8
            assert mask.shape == baseline.shape
        self.mask = mask

        if adjustments is None:
            adjustments = DataFrame(
                index=DatetimeIndex([]),
                columns=ADJUSTMENT_COLUMNS,
            )
        else:
            # Ensure that columns are in the correct order.
            adjustments = adjustments.reindex_axis(ADJUSTMENT_COLUMNS, axis=1)
            adjustments.sort(['apply_date', 'sid'], inplace=True)

        self.adjustments = adjustments
        self.adjustment_apply_dates = DatetimeIndex(adjustments.apply_date)
        self.adjustment_sids = Int64Index(adjustments.sid)

    def format_adjustments(self, dates, assets):
        """
        Build a dict of Adjustment objects in the format expected by
        adjusted_array.

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
        # TODO: Consider porting this to Cython.
        if len(self.adjustments) == 0:
            return {}

        # Mask for adjustments whose apply_dates are in the requested window of
        # dates.
        date_bounds = self.adjustment_apply_dates.slice_indexer(
            dates[0],
            dates[-1],
        )
        dates_filter = zeros(len(self.adjustments), dtype='bool')
        dates_filter[date_bounds] = True

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
                ADJUSTMENT_CONSTRUCTORS[kind](
                    dates,
                    assets,
                    start_date,
                    end_date,
                    sid,
                    value,
                ),
            )
        return out

    def load_adjusted_array(self, columns, dates, assets):
        if len(columns) != 1:
            raise ValueError(
                "Can't load multiple columns with DataFrameLoader"
            )
        elif columns[0] != self.column:
            raise ValueError("Can't load unknown column %s" % columns[0])

        # Get the indices of rows/columns corresponding to the requested
        # dates/assets.  These will have -1 on unmatched entries.
        date_indexer = self.dates.get_indexer(dates)
        assets_indexer = self.assets.get_indexer(assets)
        dates_and_assets_indexer = ix_(date_indexer, assets_indexer)

        # Select out the rows/columns that were matched.  The unmatched
        # rows/columns will be filled with garbage values. (They'll be equal to
        # the last row/column).
        baseline = self.baseline[dates_and_assets_indexer]

        # Boolean arrays with True on unmatched entries.
        missing_dates = (date_indexer == -1)
        missing_assets = (assets_indexer == -1)

        if missing_dates.any() or missing_assets.any():
            base_mask = self.mask or zeros_like(baseline, dtype=uint8)
            mask = base_mask[dates_and_assets_indexer]

            # This is a broadcasting OR between the row and column indexers.
            # e.g.
            # In [4]: missing_rows = np.array([True, False, True])
            # In [5]: missing_columns = np.array([True, False, True])
            # In [6]: missing_rows[:, None] | missing_columns
            # Out[6]: array([[ True,  True,  True],
            #                [ True, False,  True],
            #                [ True,  True,  True]], dtype=bool)
            missing_indexer = missing_dates[:, None] | missing_assets

            # Mark as bad all entries that are in a bad row OR a bad column.
            mask[missing_indexer] = 1
        else:
            mask = None

        return adjusted_array(
            data=baseline,
            mask=mask,
            adjustments=self.format_adjustments(dates, assets)
        )
