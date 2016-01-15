from .core import (
    TS_FIELD_NAME,
    SID_FIELD_NAME,
)
from zipline.pipeline.data import (CashBuybackAuthorizations,
                                   ShareBuybackAuthorizations)
from zipline.pipeline.loaders.buyback_auth import (
    CashBuybackAuthorizationsLoader,
    ShareBuybackAuthorizationsLoader
)
from .events import BlazeEventsCalendarLoader


BUYBACK_ANNOUNCEMENT_FIELD_NAME = 'buyback_dates'
SHARE_COUNT_FIELD_NAME = 'share_counts'
VALUE_FIELD_NAME = 'values'


class BlazeCashBuybackAuthorizationsLoader(BlazeEventsCalendarLoader):
    """A pipeline loader for the ``BuybackAuth`` dataset that loads
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
           {BUYBACK_ANNOUNCEMENT_FIELD_NAME}: ?datetime,
           {VALUE_FIELD_NAME}: ?float64
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
        VALUE_FIELD_NAME=VALUE_FIELD_NAME
    )

    _expected_fields = frozenset({
        TS_FIELD_NAME,
        SID_FIELD_NAME,
        BUYBACK_ANNOUNCEMENT_FIELD_NAME,
        VALUE_FIELD_NAME
    })

    def __init__(self,
                 expr,
                 dataset=CashBuybackAuthorizations,
                 loader=CashBuybackAuthorizationsLoader,
                 **kwargs):
        super(
            BlazeCashBuybackAuthorizationsLoader, self
        ).__init__(expr, dataset=dataset, loader=loader, **kwargs)


class BlazeShareBuybackAuthorizationsLoader(BlazeEventsCalendarLoader):
    """A pipeline loader for the ``BuybackAuth`` dataset that loads
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
                 dataset=ShareBuybackAuthorizations,
                 loader=ShareBuybackAuthorizationsLoader,
                 **kwargs):
        super(
            BlazeShareBuybackAuthorizationsLoader, self
        ).__init__(expr, dataset=dataset, loader=loader, **kwargs)
