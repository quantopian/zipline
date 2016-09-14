from zipline.pipeline.common import SID_FIELD_NAME, TS_FIELD_NAME
from zipline.pipeline.loaders.blaze.core import ffill_query_in_range
from zipline.pipeline.loaders.utils import (
    normalize_data_query_bounds,
    normalize_timestamp_to_query_time,
)


def load_raw_data(assets,
                  dates,
                  data_query_time,
                  data_query_tz,
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
    dates : pd.datetimeindex
        the simulation dates to load data for.
    data_query_time : datetime.time
        the time used as cutoff for new information.
    data_query_tz : tzinfo
        the timezone to normalize your dates to before comparing against
        `time`.
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
    lower_dt, upper_dt = normalize_data_query_bounds(
        dates[0],
        dates[-1],
        data_query_time,
        data_query_tz,
    )
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
    if data_query_time is not None:
        normalize_timestamp_to_query_time(
            raw,
            data_query_time,
            data_query_tz,
            inplace=True,
            ts_field=TS_FIELD_NAME,
        )
    return raw
