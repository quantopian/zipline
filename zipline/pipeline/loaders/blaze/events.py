from datashape import istabular

from .core import (
    bind_expression_to_resources,
)
from zipline.pipeline.common import SID_FIELD_NAME, TS_FIELD_NAME, \
    EVENT_DATE_FIELD_NAME
from zipline.pipeline.loaders.base import PipelineLoader
from zipline.pipeline.loaders.blaze.utils import load_raw_data
from zipline.pipeline.loaders.events import (
    EventsLoader,
    required_event_fields,
)
from zipline.pipeline.loaders.utils import (
    check_data_query_args,
)
from zipline.utils.input_validation import ensure_timezone, optionally
from zipline.utils.preprocess import preprocess


class BlazeEventsLoader(PipelineLoader):
    """An abstract pipeline loader for the events datasets that loads
    data from a blaze expression.

    Parameters
    ----------
    expr : Expr
        The expression representing the data to load.
    next_value_columns : dict[BoundColumn -> raw column name]
        A dict mapping 'next' BoundColumns to their column names in `expr`.
    previous_value_columns : dict[BoundColumn -> raw column name]
        A dict mapping 'previous' BoundColumns to their column names in `expr`.
    resources : dict, optional
        Mapping from the loadable terms of ``expr`` to actual data resources.
    odo_kwargs : dict, optional
        Extra keyword arguments to pass to odo when executing the expression.
    data_query_time : time, optional
        The time to use for the data query cutoff.
    data_query_tz : tzinfo or str
        The timezone to use for the data query cutoff.

    Notes
    -----
    The expression should have a tabular dshape of::

       Dim * {{
           {SID_FIELD_NAME}: int64,
           {TS_FIELD_NAME}: datetime,
           {EVENT_DATE_FIELD_NAME}: datetime,
       }}

    And other dataset-specific fields, where each row of the table is a
    record including the sid to identify the company, the timestamp where we
    learned about the announcement, and the event date.

    If the '{TS_FIELD_NAME}' field is not included it is assumed that we
    start the backtest with knowledge of all announcements.
    """

    __doc__ = __doc__.format(SID_FIELD_NAME=SID_FIELD_NAME,
                             TS_FIELD_NAME=TS_FIELD_NAME,
                             EVENT_DATE_FIELD_NAME=EVENT_DATE_FIELD_NAME)

    @preprocess(data_query_tz=optionally(ensure_timezone))
    def __init__(self,
                 expr,
                 next_value_columns,
                 previous_value_columns,
                 resources=None,
                 odo_kwargs=None,
                 data_query_time=None,
                 data_query_tz=None):

        dshape = expr.dshape
        if not istabular(dshape):
            raise ValueError(
                'expression dshape must be tabular, got: %s' % dshape,
            )

        required_cols = list(
            required_event_fields(next_value_columns, previous_value_columns)
        )
        self._expr = bind_expression_to_resources(
            expr[required_cols],
            resources,
        )
        self._next_value_columns = next_value_columns
        self._previous_value_columns = previous_value_columns
        self._odo_kwargs = odo_kwargs if odo_kwargs is not None else {}
        check_data_query_args(data_query_time, data_query_tz)
        self._data_query_time = data_query_time
        self._data_query_tz = data_query_tz

    def load_adjusted_array(self, columns, dates, assets, mask):
        raw = load_raw_data(assets,
                            dates,
                            self._data_query_time,
                            self._data_query_tz,
                            self._expr,
                            self._odo_kwargs)

        return EventsLoader(
            events=raw,
            next_value_columns=self._next_value_columns,
            previous_value_columns=self._previous_value_columns,
        ).load_adjusted_array(
            columns,
            dates,
            assets,
            mask,
        )
