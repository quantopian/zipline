from zipline.pipeline.common import (
    COUNT_FIELD_NAME,
    FISCAL_QUARTER_FIELD_NAME,
    FISCAL_YEAR_FIELD_NAME,
    HIGH_FIELD_NAME,
    LOW_FIELD_NAME,
    MEAN_FIELD_NAME,
    RELEASE_DATE_FIELD_NAME,
    SID_FIELD_NAME,
    STANDARD_DEVIATION_FIELD_NAME,
    TS_FIELD_NAME,
    ACTUAL_VALUE_FIELD_NAME
)
from zipline.pipeline.data import ConsensusEstimates
from zipline.pipeline.loaders import ConsensusEstimatesLoader
from .events import BlazeEventsLoader


class BlazeConsensusEstimatesLoader(BlazeEventsLoader):
    """A pipeline loader for the ``ConsensusEstimates`` dataset that
    loads
    data from a blaze expression.

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
           {RELEASE_DATE_FIELD_NAME}: ?datetime,
           {STANDARD_DEVIATION_FIELD_NAME}: ?float64,
           {COUNT_FIELD_NAME}: ?float64,
           {FISCAL_QUARTER_FIELD_NAME}: ?float64,
           {HIGH_FIELD_NAME}: ?float64,
           {MEAN_FIELD_NAME}: ?float64,
           {FISCAL_YEAR_FIELD_NAME}: ?float64,
           {LOW_FIELD_NAME}: ?float64,
           {ACTUAL_VALUE_FIELD_NAME}: ?float64
       }}

    Where each row of the table is a record including the sid to identify the
    company, the timestamp where we learned about the announcement,
    the release date for the corresponding estimate, and other estimate
    information.

    If the '{TS_FIELD_NAME}' field is not included it is assumed that we
    start the backtest with knowledge of all announcements.
    """

    __doc__ = __doc__.format(
        TS_FIELD_NAME=TS_FIELD_NAME,
        SID_FIELD_NAME=SID_FIELD_NAME,
        RELEASE_DATE_FIELD_NAME=RELEASE_DATE_FIELD_NAME,
        STANDARD_DEVIATION_FIELD_NAME=STANDARD_DEVIATION_FIELD_NAME,
        COUNT_FIELD_NAME=COUNT_FIELD_NAME,
        FISCAL_QUARTER_FIELD_NAME=FISCAL_QUARTER_FIELD_NAME,
        HIGH_FIELD_NAME=HIGH_FIELD_NAME,
        MEAN_FIELD_NAME=MEAN_FIELD_NAME,
        FISCAL_YEAR_FIELD_NAME=FISCAL_YEAR_FIELD_NAME,
        LOW_FIELD_NAME=LOW_FIELD_NAME,
        ACTUAL_VALUE_FIELD_NAME=ACTUAL_VALUE_FIELD_NAME
    )

    _expected_fields = frozenset({
        TS_FIELD_NAME,
        SID_FIELD_NAME,
        RELEASE_DATE_FIELD_NAME,
        STANDARD_DEVIATION_FIELD_NAME,
        COUNT_FIELD_NAME,
        FISCAL_QUARTER_FIELD_NAME,
        HIGH_FIELD_NAME,
        MEAN_FIELD_NAME,
        FISCAL_YEAR_FIELD_NAME,
        LOW_FIELD_NAME,
        ACTUAL_VALUE_FIELD_NAME
    })

    concrete_loader = ConsensusEstimatesLoader
    default_dataset = ConsensusEstimates
