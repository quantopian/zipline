from datashape import istabular

from .core import (
    bind_expression_to_resources,
)
from zipline.pipeline.loaders.base import PipelineLoader
from zipline.pipeline.loaders.blaze.utils import load_raw_data
from zipline.pipeline.loaders.quarter_estimates import (
    NextQuartersEstimatesLoader,
    PreviousQuartersEstimatesLoader,
    required_estimates_fields,
)
from zipline.pipeline.loaders.utils import (
    check_data_query_args,
)
from zipline.utils.input_validation import ensure_timezone, optionally
from zipline.utils.preprocess import preprocess


class BlazeEstimatesLoader(PipelineLoader):
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
    data_query_time : time, optional
        The time to use for the data query cutoff.
    data_query_tz : tzinfo or str
        The timezeone to use for the data query cutoff.
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

    @preprocess(data_query_tz=optionally(ensure_timezone))
    def __init__(self,
                 expr,
                 columns,
                 resources=None,
                 odo_kwargs=None,
                 data_query_time=None,
                 data_query_tz=None,
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
        check_data_query_args(data_query_time, data_query_tz)
        self._data_query_time = data_query_time
        self._data_query_tz = data_query_tz
        self._checkpoints = checkpoints

    def load_adjusted_array(self, columns, dates, assets, mask):
        raw = load_raw_data(assets,
                            dates,
                            self._data_query_time,
                            self._data_query_tz,
                            self._expr,
                            self._odo_kwargs,
                            checkpoints=self._checkpoints)

        return self.loader(
            raw,
            self._columns,
        ).load_adjusted_array(
            columns,
            dates,
            assets,
            mask,
        )


class BlazeNextEstimatesLoader(BlazeEstimatesLoader):
    loader = NextQuartersEstimatesLoader


class BlazePreviousEstimatesLoader(BlazeEstimatesLoader):
    loader = PreviousQuartersEstimatesLoader
