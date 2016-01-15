from .core import (
    ANNOUNCEMENT_FIELD_NAME,
    TS_FIELD_NAME,
    SID_FIELD_NAME
)
from zipline.pipeline.data import EarningsCalendar
from zipline.pipeline.loaders.earnings import EarningsCalendarLoader
from .events import BlazeEventsCalendarLoader


class BlazeEarningsCalendarLoader(BlazeEventsCalendarLoader):
    """A pipeline loader for the ``EarningsCalendar`` dataset that loads
    data from a blaze expression.

    Parameters
    ----------
    expr : Expr
        The expression representing the data to load.
    resources : dict, optional
        Mapping from the atomic terms of ``expr`` to actual data resources.
    odo_kwargs : dict, optional
        Extra keyword arguments to pass to odo when executing the expression.
    data_query_time : time, optional
        The time to use for the data query cutoff.
    data_query_tz : tzinfo or str
        The timezeone to use for the data query cutoff.

    Notes
    -----
    The expression should have a tabular dshape of::

       Dim * {{
           {SID_FIELD_NAME}: int64,
           {TS_FIELD_NAME}: datetime,
           {ANNOUNCEMENT_FIELD_NAME}: ?datetime,
       }}

    Where each row of the table is a record including the sid to identify the
    company, the timestamp where we learned about the announcement, and the
    date when the earnings will be announced.

    If the '{TS_FIELD_NAME}' field is not included it is assumed that we
    start the backtest with knowledge of all announcements.
    """

    __doc__ = __doc__.format(
        TS_FIELD_NAME=TS_FIELD_NAME,
        SID_FIELD_NAME=SID_FIELD_NAME,
        ANNOUNCEMENT_FIELD_NAME=ANNOUNCEMENT_FIELD_NAME,
    )

    _expected_fields = frozenset({
        TS_FIELD_NAME,
        SID_FIELD_NAME,
        ANNOUNCEMENT_FIELD_NAME,
    })

    def __init__(self,
                 expr,
                 resources=None,
                 odo_kwargs=None,
                 data_query_time=None,
                 data_query_tz=None,
                 dataset=EarningsCalendar,
                 loader=EarningsCalendarLoader,
                 **kwargs):
        super(
            BlazeEarningsCalendarLoader, self
        ).__init__(expr, dataset=dataset, loader=loader, resources=resources,
                   odo_kwargs=odo_kwargs, data_query_time=data_query_time,
                   data_query_tz=data_query_tz, **kwargs)
