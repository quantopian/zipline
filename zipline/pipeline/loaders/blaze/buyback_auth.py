from .core import (
    TS_FIELD_NAME,
    SID_FIELD_NAME,
)
from zipline.pipeline.data import (CashBuybackAuthorizations,
                                   ShareBuybackAuthorizations)
from zipline.pipeline.loaders.buyback_auth import (
    BUYBACK_ANNOUNCEMENT_FIELD_NAME,
    CashBuybackAuthorizationsLoader,
    CASH_FIELD_NAME,
    ShareBuybackAuthorizationsLoader,
    SHARE_COUNT_FIELD_NAME
)
from .events import BlazeEventsCalendarLoader


class BlazeCashBuybackAuthorizationsLoader(BlazeEventsCalendarLoader):
    """A pipeline loader for the ``CashBuybackAuthorizations`` dataset that loads
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
    dataset: DataSet
        The DataSet object for which this loader loads data.
    loader: EventsLoader
        The reference loader to use for this dataset.

    Notes
    -----
    The expression should have a tabular dshape of::

       Dim * {{
           {SID_FIELD_NAME}: int64,
           {TS_FIELD_NAME}: datetime,
           {BUYBACK_ANNOUNCEMENT_FIELD_NAME}: ?datetime,
           {CASH_FIELD_NAME}: ?float64
       }}

    Where each row of the table is a record including the sid to identify the
    company, the timestamp where we learned about the announcement, the
    date when the buyback was announced, the share count, and the cash amount.

    If the '{TS_FIELD_NAME}' field is not included it is assumed that we
    start the backtest with knowledge of all announcements.
    """
    __doc__ = __doc__.format(
        TS_FIELD_NAME=TS_FIELD_NAME,
        SID_FIELD_NAME=SID_FIELD_NAME,
        BUYBACK_ANNOUNCEMENT_FIELD_NAME=BUYBACK_ANNOUNCEMENT_FIELD_NAME,
        CASH_FIELD_NAME=CASH_FIELD_NAME
    )

    _expected_fields = frozenset({
        TS_FIELD_NAME,
        SID_FIELD_NAME,
        BUYBACK_ANNOUNCEMENT_FIELD_NAME,
        CASH_FIELD_NAME
    })

    def __init__(self,
                 expr,
                 resources=None,
                 odo_kwargs=None,
                 data_query_time=None,
                 data_query_tz=None,
                 dataset=CashBuybackAuthorizations,
                 loader=CashBuybackAuthorizationsLoader,
                 **kwargs):
        super(
            BlazeCashBuybackAuthorizationsLoader, self
        ).__init__(expr,
                   resources=resources,
                   odo_kwargs=odo_kwargs,
                   data_query_time=data_query_time,
                   data_query_tz=data_query_tz,
                   dataset=dataset,
                   loader=loader,
                   **kwargs)


class BlazeShareBuybackAuthorizationsLoader(BlazeEventsCalendarLoader):
    """A pipeline loader for the ``ShareBuybackAuthorizations`` dataset that loads
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
    dataset: DataSet
        The DataSet object for which this loader loads data.
    loader: EventsLoader
        The reference loader to use for this dataset.

    Notes
    -----
    The expression should have a tabular dshape of::

       Dim * {{
           {SID_FIELD_NAME}: int64,
           {TS_FIELD_NAME}: datetime,
           {BUYBACK_ANNOUNCEMENT_FIELD_NAME}: ?datetime,
           {SHARE_COUNT_FIELD_NAME}: ?float64,
       }}

    Where each row of the table is a record including the sid to identify the
    company, the timestamp where we learned about the announcement, the
    date when the buyback was announced, the share count, and the value.

    If the '{TS_FIELD_NAME}' field is not included it is assumed that we
    start the backtest with knowledge of all announcements.
    """
    __doc__ = __doc__.format(
        TS_FIELD_NAME=TS_FIELD_NAME,
        SID_FIELD_NAME=SID_FIELD_NAME,
        BUYBACK_ANNOUNCEMENT_FIELD_NAME=BUYBACK_ANNOUNCEMENT_FIELD_NAME,
        SHARE_COUNT_FIELD_NAME=SHARE_COUNT_FIELD_NAME,
    )

    _expected_fields = frozenset({
        TS_FIELD_NAME,
        SID_FIELD_NAME,
        BUYBACK_ANNOUNCEMENT_FIELD_NAME,
        SHARE_COUNT_FIELD_NAME,
    })

    def __init__(self,
                 expr,
                 resources=None,
                 odo_kwargs=None,
                 data_query_time=None,
                 data_query_tz=None,
                 dataset=ShareBuybackAuthorizations,
                 loader=ShareBuybackAuthorizationsLoader,
                 **kwargs):
        super(
            BlazeShareBuybackAuthorizationsLoader, self
        ).__init__(expr,
                   resources=resources,
                   odo_kwargs=odo_kwargs,
                   data_query_time=data_query_time,
                   data_query_tz=data_query_tz,
                   dataset=dataset,
                   loader=loader,
                   **kwargs)
