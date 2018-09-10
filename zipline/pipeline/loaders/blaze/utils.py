from zipline.pipeline.common import SID_FIELD_NAME
from zipline.pipeline.loaders.blaze.core import ffill_query_in_range


def load_raw_data(assets,
                  data_query_cutoff_times,
                  expr,
                  odo_kwargs,
                  checkpoints=None):
    """
    Given an expression representing data to load, perform normalization and
    forward-filling and return the data, materialized. Only accepts data with a
    `sid` field.

    Parameters
    ----------
    assets : pd.int64index
        the assets to load data for.
    data_query_cutoff_times : pd.DatetimeIndex
        The datetime when data should no longer be considered available for
        a session.
    expr : expr
        the expression representing the data to load.
    odo_kwargs : dict
        extra keyword arguments to pass to odo when executing the expression.
    checkpoints : expr, optional
        the expression representing the checkpointed data for `expr`.

    Returns
    -------
    raw : pd.dataframe
        The result of computing expr and materializing the result as a
        dataframe.
    """
    lower_dt, upper_dt = data_query_cutoff_times[[0, -1]]
    raw = ffill_query_in_range(
        expr,
        lower_dt,
        upper_dt,
        checkpoints=checkpoints,
        odo_kwargs=odo_kwargs,
    )
    sids = raw[SID_FIELD_NAME]
    raw.drop(
        sids[~sids.isin(assets)].index,
        inplace=True
    )
    return raw
