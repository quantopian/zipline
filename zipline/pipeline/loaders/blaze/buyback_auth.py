from .core import (
    SID_FIELD_NAME,
    TS_FIELD_NAME,
)
from zipline.pipeline.common import (
    BUYBACK_AMOUNT_FIELD_NAME,
    BUYBACK_ANNOUNCEMENT_FIELD_NAME,
    BUYBACK_TYPE_FIELD_NAME,
    BUYBACK_UNIT_FIELD_NAME,
)
from zipline.pipeline.data import BuybackAuthorizations
from zipline.pipeline.loaders import BuybackAuthorizationsLoader
from .events import BlazeEventsLoader


class BlazeBuybackAuthorizationsLoader(BlazeEventsLoader):
    """A pipeline loader for the ``BuybackAuthorizations`` dataset that
    loads data from a blaze expression.

    Parameters
    ----------
    expr : Expr
        The expression representing the data to load.
    resources : dict, optional
        Mapping from the loadable terms of ``expr`` to actual data resources.
    odo_kwargs : dict, optional
        Extra keyword arguments to pass to odo when executing the expression.
    data_query_time : time, optional
        The time to use for the data query cutoff.
    data_query_tz : tzinfo or str
        The timezeone to use for the data query cutoff.
    dataset: DataSet
        The DataSet object for which this loader loads data.

    Notes
    -----
    The expression should have a tabular dshape of::

       Dim * {{
           {SID_FIELD_NAME}: int64,
           {TS_FIELD_NAME}: datetime,
           {BUYBACK_ANNOUNCEMENT_FIELD_NAME}: ?datetime,
           {BUYBACK_AMOUNT_FIELD_NAME}: ?float64,
           {BUYBACK_UNIT_FIELD_NAME}: ?str,
           {BUYBACK_TYPE_FIELD_NAME}: ?str,
       }}

    Where each row of the table is a record including the sid to identify the
    company, the timestamp where we learned about the announcement, the
    date when the buyback was announced, the buyback amount, the buyback unit,
     and the buyback type.

    If the '{TS_FIELD_NAME}' field is not included it is assumed that we
    start the backtest with knowledge of all announcements.
    """
    __doc__ = __doc__.format(
        TS_FIELD_NAME=TS_FIELD_NAME,
        SID_FIELD_NAME=SID_FIELD_NAME,
        BUYBACK_ANNOUNCEMENT_FIELD_NAME=BUYBACK_ANNOUNCEMENT_FIELD_NAME,
        BUYBACK_AMOUNT_FIELD_NAME=BUYBACK_AMOUNT_FIELD_NAME,
        BUYBACK_UNIT_FIELD_NAME=BUYBACK_UNIT_FIELD_NAME,
        BUYBACK_TYPE_FIELD_NAME=BUYBACK_TYPE_FIELD_NAME
    )

    _expected_fields = frozenset({
        TS_FIELD_NAME,
        SID_FIELD_NAME,
        BUYBACK_ANNOUNCEMENT_FIELD_NAME,
        BUYBACK_AMOUNT_FIELD_NAME,
        BUYBACK_UNIT_FIELD_NAME,
        BUYBACK_TYPE_FIELD_NAME
    })

    concrete_loader = BuybackAuthorizationsLoader
    default_dataset = BuybackAuthorizations
