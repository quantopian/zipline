import abc

from datashape import istabular

from .core import (
    bind_expression_to_resources,
    ffill_query_in_range,
)
from zipline.pipeline.loaders.base import PipelineLoader
from zipline.pipeline.common import (
    SID_FIELD_NAME,
    TS_FIELD_NAME,
)
from zipline.pipeline.loaders.utils import (
    check_data_query_args,
    normalize_data_query_bounds,
    normalize_timestamp_to_query_time,
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
    resources : dict, optional
        Mapping from the loadable terms of ``expr`` to actual data resources.
    odo_kwargs : dict, optional
        Extra keyword arguments to pass to odo when executing the expression.
    data_query_time : time, optional
        The time to use for the data query cutoff.
    data_query_tz : tzinfo or str
        The timezeone to use for the data query cutoff.
    dataset : DataSet
        The DataSet object for which this loader loads data.

    Notes
    -----
    The expression should have a tabular dshape of::

       Dim * {{
           {SID_FIELD_NAME}: int64,
           {TS_FIELD_NAME}: datetime,
       }}

    And other dataset-specific fields, where each row of the table is a
    record including the sid to identify the company, the timestamp where we
    learned about the announcement, and the date when the earnings will be z
    announced.

    If the '{TS_FIELD_NAME}' field is not included it is assumed that we
    start the backtest with knowledge of all announcements.
    """
    default_dataset = None

    @preprocess(data_query_tz=optionally(ensure_timezone))
    def __init__(self,
                 expr,
                 resources=None,
                 odo_kwargs=None,
                 data_query_time=None,
                 data_query_tz=None,
                 dataset=default_dataset):
        if dataset is None:
            dataset = self.default_dataset

        dshape = expr.dshape

        if not istabular(dshape):
            raise ValueError(
                'expression dshape must be tabular, got: %s' % dshape,
            )

        expected_fields = self._expected_fields
        self._expr = bind_expression_to_resources(
            expr[list(expected_fields)],
            resources,
        )
        self._odo_kwargs = odo_kwargs if odo_kwargs is not None else {}
        self._dataset = dataset
        check_data_query_args(data_query_time, data_query_tz)
        self._data_query_time = data_query_time
        self._data_query_tz = data_query_tz

    @abc.abstractproperty
    def concrete_loader(self):
        NotImplementedError('concrete_loader')

    def load_adjusted_array(self, columns, dates, assets, mask):
        data_query_time = self._data_query_time
        data_query_tz = self._data_query_tz
        lower_dt, upper_dt = normalize_data_query_bounds(
            dates[0],
            dates[-1],
            data_query_time,
            data_query_tz,
        )

        raw = ffill_query_in_range(
            self._expr,
            lower_dt,
            upper_dt,
            self._odo_kwargs,
        )
        sids = raw.loc[:, SID_FIELD_NAME]
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
        gb = raw.groupby(SID_FIELD_NAME)
        return self.concrete_loader(
            dates,
            self.prepare_data(raw, gb),
            dataset=self._dataset,
        ).load_adjusted_array(columns, dates, assets, mask)

    def prepare_data(self, raw, gb):
        return {sid: raw.loc[group].drop(SID_FIELD_NAME, axis=1) for sid, group
                in gb.groups.items()}
