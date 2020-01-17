from interface import implements
from datashape import istabular

from .core import (
    bind_expression_to_resources,
)
from zipline.pipeline.common import (
    EVENT_DATE_FIELD_NAME,
    FISCAL_QUARTER_FIELD_NAME,
    FISCAL_YEAR_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME,
)
from zipline.pipeline.loaders.base import PipelineLoader
from zipline.pipeline.loaders.blaze.utils import load_raw_data
from zipline.pipeline.loaders.earnings_estimates import (
    NextEarningsEstimatesLoader,
    PreviousEarningsEstimatesLoader,
    required_estimates_fields,
    metadata_columns,
    PreviousSplitAdjustedEarningsEstimatesLoader,
    NextSplitAdjustedEarningsEstimatesLoader,
)


class BlazeEstimatesLoader(implements(PipelineLoader)):
    """An abstract pipeline loader for the estimates datasets that loads
    data from a blaze expression.

    Parameters
    ----------
    expr : Expr
        The expression representing the data to load.
    columns : dict[str -> str]
        A dict mapping BoundColumn names to the associated names in `expr`.
    resources : dict, optional
        Mapping from the loadable terms of ``expr`` to actual data resources.
    odo_kwargs : dict, optional
        Extra keyword arguments to pass to odo when executing the expression.
    checkpoints : Expr, optional
        The expression representing checkpointed data to be used for faster
        forward-filling of data from `expr`.

    Notes
    -----
    The expression should have a tabular dshape of::

       Dim * {{
           {SID_FIELD_NAME}: int64,
           {TS_FIELD_NAME}: datetime,
           {FISCAL_YEAR_FIELD_NAME}: float64,
           {FISCAL_QUARTER_FIELD_NAME}: float64,
           {EVENT_DATE_FIELD_NAME}: datetime,
       }}

    And other dataset-specific fields, where each row of the table is a
    record including the sid to identify the company, the timestamp where we
    learned about the announcement, and the date of the event.

    If the '{TS_FIELD_NAME}' field is not included it is assumed that we
    start the backtest with knowledge of all announcements.
    """
    __doc__ = __doc__.format(
        SID_FIELD_NAME=SID_FIELD_NAME,
        TS_FIELD_NAME=TS_FIELD_NAME,
        FISCAL_YEAR_FIELD_NAME=FISCAL_YEAR_FIELD_NAME,
        FISCAL_QUARTER_FIELD_NAME=FISCAL_QUARTER_FIELD_NAME,
        EVENT_DATE_FIELD_NAME=EVENT_DATE_FIELD_NAME,
    )

    def __init__(self,
                 expr,
                 columns,
                 resources=None,
                 odo_kwargs=None,
                 checkpoints=None):

        dshape = expr.dshape
        if not istabular(dshape):
            raise ValueError(
                'expression dshape must be tabular, got: %s' % dshape,
            )

        required_cols = list(
            required_estimates_fields(columns)
        )
        self._expr = bind_expression_to_resources(
            expr[required_cols],
            resources,
        )
        self._columns = columns
        self._odo_kwargs = odo_kwargs if odo_kwargs is not None else {}
        self._checkpoints = checkpoints

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        # Only load requested columns.
        requested_column_names = [self._columns[column.name]
                                  for column in columns]

        if dates.tzinfo is not None:
            dates = dates.tz_convert("UTC").tz_localize(None)

        raw = load_raw_data(
            sids,
            dates,
            self._expr[sorted(metadata_columns.union(requested_column_names))],
            self._odo_kwargs,
            checkpoints=self._checkpoints,
        )

        return self.loader(
            raw,
            {column.name: self._columns[column.name] for column in columns},
        ).load_adjusted_array(
            domain,
            columns,
            dates,
            sids,
            mask,
        )


class BlazeNextEstimatesLoader(BlazeEstimatesLoader):
    loader = NextEarningsEstimatesLoader


class BlazePreviousEstimatesLoader(BlazeEstimatesLoader):
    loader = PreviousEarningsEstimatesLoader


class BlazeSplitAdjustedEstimatesLoader(BlazeEstimatesLoader):
    def __init__(self,
                 expr,
                 columns,
                 split_adjustments_loader,
                 split_adjusted_column_names,
                 split_adjusted_asof,
                 **kwargs):
        self._split_adjustments = split_adjustments_loader
        self._split_adjusted_column_names = split_adjusted_column_names
        self._split_adjusted_asof = split_adjusted_asof
        super(BlazeSplitAdjustedEstimatesLoader, self).__init__(
            expr,
            columns,
            **kwargs
        )

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        # Only load requested columns.
        requested_column_names = [self._columns[column.name]
                                  for column in columns]

        requested_spilt_adjusted_columns = [
            column_name
            for column_name in self._split_adjusted_column_names
            if column_name in requested_column_names
        ]

        if dates.tzinfo is not None:
            dates = dates.tz_convert("UTC").tz_localize(None)

        domain_dates = domain.data_query_cutoff_for_sessions(dates).tz_convert("UTC").tz_localize(None)

        raw = load_raw_data(
            sids,
            domain.data_query_cutoff_for_sessions(dates),
            self._expr[sorted(metadata_columns.union(requested_column_names))],
            self._odo_kwargs,
            checkpoints=self._checkpoints,
        )

        return self.loader(
            raw,
            {column.name: self._columns[column.name] for column in columns},
            self._split_adjustments,
            requested_spilt_adjusted_columns,
            self._split_adjusted_asof,
        ).load_adjusted_array(
            domain,
            columns,
            dates,
            sids,
            mask,
        )


class BlazeNextSplitAdjustedEstimatesLoader(BlazeSplitAdjustedEstimatesLoader):
    loader = NextSplitAdjustedEarningsEstimatesLoader


class BlazePreviousSplitAdjustedEstimatesLoader(
    BlazeSplitAdjustedEstimatesLoader
):
    loader = PreviousSplitAdjustedEarningsEstimatesLoader
