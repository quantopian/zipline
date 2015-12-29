import blaze as bz
from datashape import istabular
from odo import odo
import pandas as pd
from six import iteritems
from toolz import valmap

from .core import TS_FIELD_NAME, SID_FIELD_NAME
from zipline.pipeline.data import EarningsCalendar
from zipline.pipeline.loaders.base import PipelineLoader
from zipline.pipeline.loaders.earnings import EarningsCalendarLoader


ANNOUNCEMENT_FIELD_NAME = 'announcement_date'


def bind_expression_to_resources(expr, resources):
    """
    Bind a Blaze expression to resources.

    Parameters
    ----------
    expr : bz.Expr
        The expression to which we want to bind resources.
    resources : dict[bz.Symbol -> any]
        Mapping from the atomic terms of ``expr`` to actual data resources.

    Returns
    -------
    bound_expr : bz.Expr
        ``expr`` with bound resources.
    """
    # bind the resources into the expression
    if resources is None:
        resources = {}

    # _subs stands for substitute.  It's not actually private, blaze just
    # prefixes symbol-manipulation methods with underscores to prevent
    # collisions with data column names.
    return expr._subs({
        k: bz.Data(v, dshape=k.dshape) for k, v in iteritems(resources)
    })


class BlazeEarningsCalendarLoader(PipelineLoader):
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
                 odo_kwargs=None,
                 dataset=EarningsCalendar):
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
        if pd.isnull(lower):
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

        def mkseries(idx, raw_loc=raw.loc):
            vs = raw_loc[
                idx, [TS_FIELD_NAME, ANNOUNCEMENT_FIELD_NAME]
            ].values
            return pd.Series(
                index=pd.DatetimeIndex(vs[:, 0]),
                data=vs[:, 1],
            )

        return EarningsCalendarLoader(
            dates,
            valmap(mkseries, gb.groups),
            dataset=self._dataset,
        ).load_adjusted_array(columns, dates, assets, mask)
