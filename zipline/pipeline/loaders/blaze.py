"""Blaze integration with the Pipeline API.
"""
from __future__ import division, absolute_import

from abc import ABCMeta, abstractproperty
from collections import namedtuple, defaultdict
from itertools import count
from operator import attrgetter
import warnings
from weakref import WeakKeyDictionary

import blaze as bz
from datashape import (
    Date,
    DateTime,
    Option,
    float64,
    isrecord,
    isscalar,
    promote,
)
from numpy.lib.stride_tricks import as_strided
from odo import odo
import pandas as pd
from toolz import flip, memoize, compose, complement, identity
from six import with_metaclass


from ..data.dataset import DataSet, Column
from zipline.lib.adjusted_array import adjusted_array
from zipline.lib.adjustment import Float64Overwrite


AD_FIELD_NAME = 'asof_date'
TS_FIELD_NAME = 'timestamp'
SID_FIELD_NAME = 'sid'
valid_deltas_node_types = (
    bz.expr.Field,
    bz.expr.ReLabel,
    bz.expr.Symbol,
)
is_invalid_deltas_node = complement(flip(isinstance, valid_deltas_node_types))
getname = attrgetter('__name__')


class _ExprRepr(object):
    """Box for repring expressions with the str of the expression.

    Parameters
    ----------
    expr : Expr
        The expression to box for repring.
    """
    __slots__ = 'expr',

    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return str(self.expr)
    __str__ = __repr__


class ExprData(namedtuple('ExprData', 'expr deltas resources')):
    """A pair of expressions and a data resources.

    Parameters
    ----------
    expr : Expr
        The first known values.
    deltas : Expr, optional
        The deltas for the data.
    resources : resource or dict of resources, optional
        The resources to compute the exprs against.
    """
    def __new__(cls, expr, deltas=None, resources=None):
        return super(ExprData, cls).__new__(cls, expr, deltas, resources)

    def __repr__(self):
        # If the expressions have _resources() then the repr will
        # drive computation so we box them.
        cls = type(self)
        return super(ExprData, cls).__repr__(cls(
            _ExprRepr(self.expr),
            _ExprRepr(self.deltas),
            self.resources,
        ))


class InvalidField(with_metaclass(ABCMeta)):
    """A field that raises an exception that indicates that the
    field was invalid.

    Parameters
    ----------
    field : str
        The name of the field.
    type_ : dshape
        The shape of the field.
    """
    @abstractproperty
    def error_format(self):
        raise NotImplementedError('error_format')

    def __init__(self, field, type_):
        self._field = field
        self._type = type_

    def __get__(self, instance, owner):
        raise AttributeError(
            self.error_format.format(field=self._field, type_=self._type),
        )


class NonNumpyField(InvalidField):
    error_format = "field '{field}' was a non numpy compatible type: '{type_}'"


class NonPipelineField(InvalidField):
    error_format = (
        "field '{field}' was a non Pipeline API compatible type:"
        " '{type_.__name__}'"
    )


class NotPipelineCompatible(TypeError):
    """Exception used to indicate that a dshape is not Pipeline API
    compatible.
    """
    def __str__(self):
        return "'%s' is a non Pipleine API compatible type'" % self.args


_new_names = ('_%d' % n for n in count())


@memoize
def new_dataset(expr, deltas):
    """Creates or returns a dataset from a pair of blaze expressions.

    Parameters
    ----------
    expr : Expr
       The blaze expression representing the first known values.
    deltas : Expr
       The blaze expression representing the deltas to the data.

    Returns
    -------
    ds : type
        A new dataset type.

    Notes
    -----
    This function is memoized, repeated calls will return the same type.
    """
    columns = {}
    for name, type_ in expr.dshape.measure.fields:
        try:
            if promote(type_, float64, promote_option=False) != float64:
                raise NotPipelineCompatible()
            if isinstance(type_, Option):
                type_ = type_.ty
        except NotPipelineCompatible:
            col = NonPipelineField(name, type_)
        except TypeError:
            col = NonNumpyField(name, type_)
        else:
            col = Column(type_.to_numpy_dtype().type)

        columns[name] = col

    name = expr._name
    if expr._name is None:
        name = next(_new_names)
    return type(name, (DataSet,), columns)


def _check_resources(name, expr, resources):
    """Validate that the exprssion and resources passed match up.

    Parameters
    ----------
    name : str
        The name of the argument we are checking.
    expr : Expr
        The potentially bound expr.
    resources
        The explicitly passed resources to compute expr.

    Raises
    ------
    ValueError
        If the resources to not match for an expression.
    """
    if expr is None:
        return
    bound = expr._resources()
    if not bound and resources is None:
        raise ValueError('no resources provided to compute %s' % name)
    if bound and resources:
        raise ValueError(
            'explicit and implicit resources provided to compute %s' % name,
        )


def _check_datetime_field(name, measure):
    """Check that a field is a datetime inside some measure.

    Parameters
    ----------
    name : str
        The name of the field to check.
    measure : Record
        The record to check the field of.

    Raises
    ------
    TypeError
        If the field is not a datetime inside ``measure``.
    """
    if not isinstance(measure[name], (Date, DateTime)):
        raise TypeError(
            "'{name}' field must be a '{dt}', not: '{dshape}'".format(
                name=name,
                dt=DateTime(),
                dshape=measure[name],
            ),
        )


class NoDeltasWarning(UserWarning):
    """Warning used to signal that no deltas could be found and none
    were provided.

    Parameters
    ----------
    expr : Expr
        The expression that was searched.
    """
    def __init__(self, expr):
        self._expr = expr

    def __str__(self):
        return 'No deltas could be infered from expr: %s' % self._expr


_valid_no_deltas_rules = 'warn', 'raise', 'ignore'


def _get_deltas(expr, deltas, no_deltas_rule):
    """Find the correct deltas for the expression.

    Parameters
    ----------
    expr : Expr
        The base expression.
    deltas : Expr, 'auto', or None
        The deltas argument. If this is 'auto', then the deltas table will
        be searched for by walking up the expression tree. If this can not be
        reflected, then an action will be taken based on the 'no_deltas_rule'.
    no_deltas_rule : {'warn', 'raise', 'ignore'}
        How to handle the case where deltas='auto' but no deltas could be
        found.

    Returns
    -------
    deltas : Expr or None
        The deltas table to use.
    """
    if no_deltas_rule not in _valid_no_deltas_rules:
        raise ValueError(
            'no_deltas_rule must be one of: %s' %
            _valid_no_deltas_rules
        )

    if isinstance(deltas, bz.Expr) or deltas != 'auto':
        return deltas

    try:
        return expr._child[(expr._name or '') + '_deltas']
    except (ValueError, AttributeError):
        if no_deltas_rule == 'raise':
            raise ValueError(
                "no deltas table could be reflected for %s" % expr
            )
        elif no_deltas_rule == 'warn':
            warnings.warn(NoDeltasWarning(expr))
    return None


def from_blaze(expr,
               deltas='auto',
               loader=None,
               resources=None,
               no_deltas_rule=_valid_no_deltas_rules[0]):
    """Create a Pipeline API object from a blaze expression.

    Parameters
    ----------
    expr : Expr
        The blaze expression to use.
    deltas : Expr or 'auto', optional
        The expression to use for the point in time adjustments.
        If the string 'auto' is passed, a deltas expr will be looked up
        by stepping up the expression tree and looking for another field
        with the name of ``expr`` + '_deltas'. If None is passed, no deltas
        will be used.
    loader : BlazeLoader, optional
        The blaze loader to attach this pipeline dataset to. If None is passed,
        the global blaze loader is used.
    resources : dict or any, optional
        The data to execute the blaze expressions against. This is used as the
        scope for ``bz.compute``.
    no_deltas_rule : {'warn', 'raise', 'ignore'}
        What should happen if ``deltas='auto'`` but no deltas can be found.
        'log' says to log a message but continue.
        'raise' says to raise an exception if no deltas can be found.
        'ignore' says take no action and proceed with no deltas.

    Returns
    -------
    pipeline_api_obj : DataSet or BoundColumn
        Either a new dataset or bound column based on the shape of the expr
        passed in. If a table shaped expression is passed, this will return
        a ``DataSet`` that represents the whole table. If an array-like shape
        is passed, a ``BoundColumn`` on the dataset that would be constructed
        from passing the parent is returned.
    """
    deltas = _get_deltas(expr, deltas, no_deltas_rule)
    if deltas is not None:
        invalid_nodes = tuple(filter(is_invalid_deltas_node, expr._subterms()))
        if invalid_nodes:
            raise TypeError(
                'expression with deltas may only contain (%s) nodes,'
                " found: %s" % (
                    ', '.join(map(getname, valid_deltas_node_types)),
                    ', '.join(map(compose(getname, type), invalid_nodes)),
                ),
            )

    # Check if this is a single column out of a dataset.
    single_column = None
    if isscalar(expr.dshape.measure):
        # This is a single column. Record which column we are to return
        # but create the entire dataset.
        single_column = expr._name
        col = expr
        for expr in expr._subterms():
            if isrecord(expr.dshape.measure):
                break
        else:
            expr = bz.Data(col, name=single_column)

    measure = expr.dshape.measure
    if not isrecord(measure) or AD_FIELD_NAME not in measure.names:
        raise TypeError(
            "expr must be a collection of records with at least an '{ad}'"
            " field. Fields provided: '{fields}'\nhint: maybe you need to use "
            " `relabel` to change your field names".format(
                ad=AD_FIELD_NAME,
                fields=measure,
            ),
        )
    _check_datetime_field(AD_FIELD_NAME, measure)

    if TS_FIELD_NAME not in measure.names:
        expr = bz.transform(expr, **{TS_FIELD_NAME: expr[AD_FIELD_NAME]})
        if deltas is not None:
            deltas = bz.transform(
                deltas,
                **{TS_FIELD_NAME: deltas[AD_FIELD_NAME]}
            )
    else:
        _check_datetime_field(TS_FIELD_NAME, measure)

    if deltas is not None and (sorted(deltas.dshape.measure.fields) !=
                               sorted(measure.fields)):
        raise TypeError(
            'base measure != deltas measure:\n%s != %s' % (
                measure,
                deltas.dshape.measure,
            ),
        )

    # Ensure that we have a data resource to execute the query against.
    _check_resources('expr', expr, resources)
    _check_resources('deltas', deltas, resources)

    # Create or retrieve the Pipeline API dataset.
    ds = new_dataset(expr, deltas)
    # Register our new dataset with the loader.
    (loader if loader is not None else global_loader)[ds] = ExprData(
        expr,
        deltas,
        resources,
    )
    if single_column is not None:
        # We were passed a single column, extract and return it.
        return getattr(ds, single_column)
    return ds


getdataset = attrgetter('dataset')
dataset_name = attrgetter('name')


def inline_novel_deltas(base, deltas, dates):
    """Inline any deltas into the base set that would have changed our most
    recently known value.

    Parameters
    ----------
    base : pd.DataFrame
        The first known values.
    deltas : pd.DataFrame
        Overwrites to the base data.
    dates : pd.DatetimeIndex
        The dates requested by the loader.

    Returns
    -------
    new_base : pd.DataFrame
        The new base data with novel deltas inserted.
    """
    get_indexes = dates.searchsorted
    return pd.concat(
        (base,
         deltas.loc[
             (get_indexes(deltas[TS_FIELD_NAME].values, 'right') -
              get_indexes(deltas[AD_FIELD_NAME].values, 'letf')) <= 1
         ].drop(AD_FIELD_NAME, 1)),
        ignore_index=True,
    )


def overwrite_from_dates(asof, dates, dense_dates, asset_idx, value):
    """Construct a `Float64Overwrite` with the correct
    start and end date based on the asof date of the delta,
    the dense_dates, and the dense_dates.

    Parameters
    ----------
    asof : datetime
        The asof date of the delta.
    dates : pd.DatetimeIndex
        The dates requested by the loader.
    dense_dates : pd.DatetimeIndex
        The dates that appeared in the dataset.
    asset_idx : int
        The index of the asset in the block.
    value : np.float64
        The value to overwrite with.

    Returns
    -------
    overwrite : Float64Overwrite
        The overwrite that will apply the new value to the data.
    """
    return Float64Overwrite(
        dates.searchsorted(asof),
        dates.get_loc(dense_dates[dense_dates.searchsorted(asof) + 1]) - 1,
        asset_idx,
        value,
    )


def adjustments_from_deltas_no_sids(dates,
                                    dense_dates,
                                    column_idx,
                                    column_name,
                                    assets,
                                    deltas):
    """Collect all the adjustments that occur in a dataset that does not
    have a sid column.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        The dates requested by the loader.
    dense_dates : pd.DatetimeIndex
        The dates that were in the dense data.
    column_idx : int
        The index of the column in the dataset.
    column_name : str
        The name of the column to compute deltas for.
    deltas : pd.DataFrame
        The overwrites that should be applied to the dataset.

    Returns
    -------
    adjustments : dict[idx -> Float64Overwrite]
        The adjustments dictionary to feed to the adjusted array.
    """
    return {
        dates.get_loc(kd): tuple(
            overwrite_from_dates(
                deltas.loc[kd, AD_FIELD_NAME],
                dates,
                dense_dates,
                n,
                v,
            ) for n in range(len(assets))
        ) for kd, v in deltas[column_name].iteritems()
    }


def adjustments_from_deltas_with_sids(dates,
                                      dense_dates,
                                      column_idx,
                                      column_name,
                                      assets,
                                      deltas):
    """Collect all the adjustments that occur in a dataset that does not
    have a sid column.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        The dates requested by the loader.
    dense_dates : pd.DatetimeIndex
        The dates that were in the dense data.
    column_idx : int
        The index of the column in the dataset.
    column_name : str
        The name of the column to compute deltas for.
    deltas : pd.DataFrame
        The overwrites that should be applied to the dataset.

    Returns
    -------
    adjustments : dict[idx -> Float64Overwrite]
        The adjustments dictionary to feed to the adjusted array.
    """
    adjustments = defaultdict(list)
    for sid_idx, (sid, per_sid) in enumerate(deltas[column_name].iteritems()):
        for kd, v in per_sid.iteritems():
            adjustments[dates.get_loc(kd)].append(
                overwrite_from_dates(
                    deltas[AD_FIELD_NAME].loc[kd, sid],
                    dates,
                    dense_dates,
                    sid_idx,
                    v,
                ),
            )
    return dict(adjustments)  # no subclasses of dict


class BlazeLoader(dict):
    def __init__(self, colmap=None):
        self.update(colmap or {})

    @classmethod
    @memoize(cache=WeakKeyDictionary())
    def global_instance(cls):
        return cls()

    def load_adjusted_array(self, columns, dates, assets, mask):
        try:
            dataset, = set(map(getdataset, columns))
        except ValueError:
            raise AssertionError('all columns must come from the same dataset')

        expr, deltas, resources = self[dataset]
        have_sids = SID_FIELD_NAME in expr.fields
        assets = list(map(int, assets))  # coerce from numpy.int64
        fields = list(map(dataset_name, columns))
        query_fields = fields + [AD_FIELD_NAME, TS_FIELD_NAME] + (
            [SID_FIELD_NAME] if have_sids else []
        )

        def where(e):
            """Create the query to run against the resources.

            Parameters
            ----------
            e : Expr
                The base or deltas expression.

            Returns
            -------
            q : Expr
                The query to run.
            """
            ts = e[TS_FIELD_NAME]
            # Hack to get the lower bound to query:
            # This must be strictly executed because the data for `ts` will
            # be removed from scope too early otherwise.
            lower = odo(ts[ts <= dates[0]].max(), pd.Timestamp)
            return e[
                (e[SID_FIELD_NAME].isin(assets) if have_sids else True) &
                ((ts >= lower) if lower is not pd.NaT else True) &
                (ts <= dates[-1])
            ][query_fields]

        materialized_expr = odo(
            bz.compute(where(expr), resources),
            pd.DataFrame,
        )

        materialized_deltas = (
            odo(bz.compute(where(deltas), resources), pd.DataFrame)
            if deltas is not None else
            pd.DataFrame(columns=query_fields)
        )
        # Inline the deltas that changed our most recently known value.
        # Also, we reindex by the dates to create a dense representation of
        # the data.
        sparse_output = inline_novel_deltas(
            materialized_expr,
            materialized_deltas,
            dates,
        ).drop(AD_FIELD_NAME, axis=1).set_index(TS_FIELD_NAME)

        if have_sids:
            # Unstack by the sid so that we get a multi-index on the columns
            # of datacolumn, sid.
            sparse_output = sparse_output.set_index(
                SID_FIELD_NAME,
                append=True,
            ).unstack()
            sparse_deltas = materialized_deltas.set_index(
                [TS_FIELD_NAME, SID_FIELD_NAME],
            ).unstack()

            # Allocate the whole output dataframe at once instead of
            # reindexing.
            dense_output = pd.DataFrame(
                columns=pd.MultiIndex.from_product(
                    (sparse_output.columns.levels[0], assets),
                    names=(
                        sparse_output.columns.levels[0].name,
                        SID_FIELD_NAME,
                    ),
                ),
                index=dates,
            )

            # In place update the output based on the base.
            dense_output.update(sparse_output)
            adjustments_from_deltas = adjustments_from_deltas_with_sids
            column_view = identity
        else:
            # We use the column view to make an array per asset.
            dense_output = sparse_output.reindex(dates)
            sparse_deltas = materialized_deltas.set_index(TS_FIELD_NAME)
            adjustments_from_deltas = adjustments_from_deltas_no_sids

            def column_view(arr, _shape=(len(dates), len(assets))):
                """Return a virtual matrix where we make a view that
                duplicates a single column for all the assets.

                Examples
                --------
                >>> arr = np.array([1, 2, 3])
                >>> as_strided(arr, shape=(3, 3), strides=(arr.itemsize, 0))
                array([[1, 1, 1],
                       [2, 2, 2],
                       [3, 3, 3]])
                """
                return as_strided(
                    arr,
                    shape=_shape,
                    strides=(arr.itemsize, 0),
                )

        # Walk forward the data after any symbol mapped or non-symbol mapped
        # specific transforms have been applied.
        sparse_output = sparse_output.ffill()

        for column_idx, column in enumerate(columns):
            column_name = column.name
            yield adjusted_array(
                column_view(
                    dense_output[column_name].values.astype(column.dtype),
                ),
                mask,
                adjustments_from_deltas(
                    dates,
                    sparse_output.index,
                    column_idx,
                    column_name,
                    assets,
                    sparse_deltas,
                )
            )

    def __repr__(self):
        return '%s(%s)' % (
            type(self).__name__,
            super(BlazeLoader, self).__repr__(),
        )

global_loader = BlazeLoader.global_instance()
