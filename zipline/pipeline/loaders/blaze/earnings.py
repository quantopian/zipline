import blaze as bz
from datashape import istabular
from odo import odo
import pandas as pd
from six import iteritems
from toolz import valmap

from .core import TS_FIELD_NAME, SID_FIELD_NAME
from zipline.pipeline.loaders.base import PipelineLoader
from zipline.pipeline.loaders.earnings import EarningsCalendarLoader


ANNOUNCEMENT_FIELD_NAME = 'announcement_date'


class BlazeEarningsCalendarLoader(PipelineLoader):
    """A pipeline loader for the ``EarningsCalendar`` dataset that loads
    data from a blaze expression.

    Parameters
    ----------
    expr : Expr
        The expression representing the data to load.
    resources : any, optional
        The resources to use when computing ``expr``. If expr is already
        bound to resources this can be omitted.
    odo_kwargs : dict, optional
        Extra keyword arguments to pass to odo when executing the expression.

    Notes
    -----
    The expression should have a tabular dshape of::

       Dim * {{
           {SID_FIELD_NAME}: int64,
           {TS_FIELD_NAME}: datetime64,
           {ANNOUNCEMENT_FIELD_NAME}: datetime64,
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
                 compute_kwargs=None,
                 odo_kwargs=None):
        dshape = expr.dshape

        if not istabular(dshape):
            raise ValueError(
                'expression dshape must be tabular, got: %s' % dshape,
            )

        expected_fields = self._expected_fields
        self._has_ts = has_ts = TS_FIELD_NAME in dshape.measure.dict
        if not has_ts:
            # This field is optional.
            expected_fields - {TS_FIELD_NAME}

        # bind the resources into the expression
        if resources is None:
            resources = {}
        elif not isinstance(resources, dict):
            leaves = expr._leaves()
            if len(leaves) != 1:
                raise ValueError('no data resources found')

            resources = {leaves[0]: resources}

        self._expr = expr[list(expected_fields)]._subs({
            k: bz.Data(v, dshape=k.dshape) for k, v in iteritems(resources)
        })
        self._odo_kwargs = odo_kwargs if odo_kwargs is not None else {}

    def load_adjusted_array(self, columns, dates, assets, mask):
        expr = self._expr
        filtered = expr[expr[TS_FIELD_NAME] <= dates[0]]
        lower = odo(
            bz.by(
                filtered[SID_FIELD_NAME],
                timestamp=filtered[TS_FIELD_NAME].max(),
            ).timestamp.min(),
            pd.Timestamp,
            **self._odo_kwargs
        )
        if lower is pd.NaT:
            # If there is no lower date, just query for data in the date
            # range. It must all be null anyways.
            lower = dates[0]

        raw = odo(
            expr[
                (expr[TS_FIELD_NAME] >= lower) &
                (expr[TS_FIELD_NAME] <= dates[-1])
            ],
            pd.DataFrame,
            **self._odo_kwargs
        )

        sids = raw.loc[:, SID_FIELD_NAME]
        raw.drop(
            sids[~(sids.isin(assets) | sids.notnull())].index,
            inplace=True
        )

        gb = raw.groupby(SID_FIELD_NAME)
        if self._has_ts:
            def mkseries(idx, raw_loc=raw.loc):
                vs = raw_loc[
                    idx, [TS_FIELD_NAME, ANNOUNCEMENT_FIELD_NAME]
                ].values
                return pd.Series(
                    index=pd.DatetimeIndex(vs[:, 0]),
                    data=vs[:, 1],
                )
        else:
            def mkseries(idx, raw_loc=raw.loc):
                return pd.DatetimeIndex(raw_loc[idx, ANNOUNCEMENT_FIELD_NAME])

        return EarningsCalendarLoader(
            dates,
            valmap(mkseries, gb.groups),
        ).load_adjusted_array(columns, dates, assets, mask)
