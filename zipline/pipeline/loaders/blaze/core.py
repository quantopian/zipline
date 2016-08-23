"""
Blaze integration with the Pipeline API.

For an overview of the blaze project, see blaze.pydata.org

The blaze loader for the Pipeline API is designed to allow us to load
data from arbitrary sources as long as we can execute the needed expressions
against the data with blaze.

Data Format
-----------

The blaze Pipeline API loader expects that data is formatted in a tabular way.
The only required column in your table is ``asof_date`` where this column
represents the date this data is referencing. For example, one might have a CSV
like:

asof_date,value
2014-01-06,0
2014-01-07,1
2014-01-08,2

This says that the value on 2014-01-01 was 0 and so on.

Optionally, we may provide a ``timestamp`` column to be used to represent
point in time data. This column tells us when the data was known, or became
available to for use. Using our same CSV, we could write this with a timestamp
like:

asof_date,timestamp,value
2014-01-06,2014-01-07,0
2014-01-07,2014-01-08,1
2014-01-08,2014-01-09,2

This says that the value was 0 on 2014-01-01; however, we did not learn this
until 2014-01-02. This is useful for avoiding look-ahead bias in your
pipelines. If this column does not exist, the ``asof_date`` column will be used
instead.

If your data references a particular asset, you can add a ``sid`` column to
your dataset to represent this. For example:

asof_date,value,sid
2014-01-06,0,10
2014-01-06,1,20
2014-01-07,1,10
2014-01-07,2,20
2014-01-08,2,10
2014-01-08,3,20

This says that on 2014-01-01, the asset with id 10 had a value of 0, and the
asset with id 20 had a value of 1.


One of the key features of the Pipeline API is the handling of adjustments and
restatements. Often our data will be amended after the fact and we would like
to trade on the newest information; however, we do not want to introduce this
knowledge to our model too early. The blaze loader handles this case by
accepting a second ``deltas`` expression that contains all of the restatements
in the original expression.

For example, let's use our table from above:

asof_date,value
2014-01-06,0
2014-01-07,1
2014-01-08,2

Imagine that on the fourth the vendor realized that the calculation was
incorrect and the value on the first was actually -1. Then, on the fifth, they
realized that the value for the third was actually 3. We can construct a
``deltas`` expression to pass to our blaze loader that has the same shape as
our baseline table but only contains these new values like:

asof_date,timestamp,value
2014-01-06,2014-01-09,-1
2014-01-08,2014-01-10,3

This shows that we learned on the fourth that the value on the first was
actually -1 and that we learned on the fifth that the value on the third was
actually 3. By pulling our data into these two tables and not silently updating
our original table we can run our pipelines using the information we would
have had on that day, and we can prevent lookahead bias in the pipelines.


Another optional expression that may be provided is ``checkpoints``. The
``checkpoints`` expression is used when doing a forward fill query to cap the
lower date that must be searched. This expression has the same shape as the
``baseline`` and ``deltas`` expressions but should be downsampled with novel
deltas applied. For example, imagine we had one data point per asset per day
for some dataset. We could dramatically speed up our queries by pre populating
a downsampled version which has the most recently known value at the start of
each month. Then, when we query, we only must look back at most one month
before the start of the pipeline query to provide enough data to forward fill
correctly.

Conversion from Blaze to the Pipeline API
-----------------------------------------

Now that our data is structured in the way that the blaze loader expects, we
are ready to convert our blaze expressions into Pipeline API objects.

This module (zipline.pipeline.loaders.blaze) exports a function called
``from_blaze`` which performs this mapping.

The expression that you are trying to convert must either be tabular or
array-like. This means the ``dshape`` must be like:

``Dim * {A: B}`` or ``Dim * A``.

This represents an expression of dimension 1 which may be fixed or variable,
whose measure is either some record or a scalar.

The record case defines the entire table with all of the columns, this maps the
blaze expression into a pipeline DataSet. This dataset will have a column for
each field of the record. Some datashape types cannot be coerced into Pipeline
API compatible types and in that case, a column cannot be constructed.
Currently any numeric type that may be promoted to a float64 is compatible with
the Pipeline API.

The scalar case defines a single column pulled out a table. For example, let
``expr = bz.symbol('s', 'var * {field: int32, asof_date: datetime}')``.
When we pass ``expr.field`` to ``from_blaze``, we will walk back up the
expression tree until we find the table that ``field`` is defined on. We will
then proceed with the record case to construct a dataset; however, before
returning the dataset we will pull out only the column that was passed in.

For full documentation, see ``help(from_blaze)`` or ``from_blaze?`` in IPython.

Using our Pipeline DataSets and Columns
---------------------------------------

Once we have mapped our blaze expressions into Pipeline API objects, we may
use them just like any other datasets or columns. For more information on how
to run a pipeline or using the Pipeline API, see:
www.quantopian.com/help#pipeline-api
"""
from __future__ import division, absolute_import

from abc import ABCMeta, abstractproperty
from collections import namedtuple, defaultdict
from functools import partial
from itertools import count
import warnings
from weakref import WeakKeyDictionary

import blaze as bz
from datashape import (
    Date,
    DateTime,
    Option,
    isrecord,
    isscalar,
)
import numpy as np
from odo import odo
import pandas as pd
from six import with_metaclass, PY2, itervalues, iteritems
from toolz import (
    complement,
    compose,
    flip,
    groupby,
    identity,
    memoize,
    merge,
)
import toolz.curried.operator as op

from zipline.pipeline.common import (
    AD_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME
)
from zipline.pipeline.data.dataset import DataSet, Column
from zipline.pipeline.loaders.utils import (
    check_data_query_args,
    last_in_date_group,
    normalize_data_query_bounds,
    normalize_timestamp_to_query_time,
    ffill_across_cols)
from zipline.pipeline.sentinels import NotSpecified
from zipline.lib.adjusted_array import AdjustedArray, can_represent_dtype
from zipline.lib.adjustment import Float64Overwrite
from zipline.utils.input_validation import (
    expect_element,
    ensure_timezone,
    optionally,
)
from zipline.utils.numpy_utils import bool_dtype
from zipline.utils.pandas_utils import sort_values
from zipline.utils.pool import SequentialPool
from zipline.utils.preprocess import preprocess


valid_deltas_node_types = (
    bz.expr.Field,
    bz.expr.ReLabel,
    bz.expr.Symbol,
)
traversable_nodes = (
    bz.expr.Field,
    bz.expr.Label,
)
is_invalid_deltas_node = complement(flip(isinstance, valid_deltas_node_types))
get__name__ = op.attrgetter('__name__')


class ExprData(namedtuple('ExprData', 'expr deltas checkpoints odo_kwargs')):
    """A pair of expressions and data resources. The expresions will be
    computed using the resources as the starting scope.

    Parameters
    ----------
    expr : Expr
        The baseline values.
    deltas : Expr, optional
        The deltas for the data.
    checkpoints : Expr, optional
        The forward fill checkpoints for the data.
    odo_kwargs : dict, optional
        The keyword arguments to forward to the odo calls internally.
    """
    def __new__(cls, expr, deltas=None, checkpoints=None, odo_kwargs=None):
        return super(ExprData, cls).__new__(
            cls,
            expr,
            deltas,
            checkpoints,
            odo_kwargs or {},
        )

    def __repr__(self):
        # If the expressions have _resources() then the repr will
        # drive computation so we take the str here.
        cls = type(self)
        return super(ExprData, cls).__repr__(cls(
            str(self.expr),
            str(self.deltas),
            str(self.checkpoints),
            self.odo_kwargs,
        ))


class InvalidField(with_metaclass(ABCMeta)):
    """A field that raises an exception indicating that the
    field was invalid.

    Parameters
    ----------
    field : str
        The name of the field.
    type_ : dshape
        The shape of the field.
    """
    @abstractproperty
    def error_format(self):  # pragma: no cover
        raise NotImplementedError('error_format')

    def __init__(self, field, type_):
        self._field = field
        self._type = type_

    def __get__(self, instance, owner):
        raise AttributeError(
            self.error_format.format(field=self._field, type_=self._type),
        )


class NonNumpyField(InvalidField):
    error_format = (
        "field '{field}' was a non numpy compatible type: '{type_}'"
    )


class NonPipelineField(InvalidField):
    error_format = (
        "field '{field}' was a non Pipeline API compatible type: '{type_}'"
    )


_new_names = ('BlazeDataSet_%d' % n for n in count())


def datashape_type_to_numpy(type_):
    """
    Given a datashape type, return the associated numpy type. Maps
    datashape's DateTime type to numpy's `datetime64[ns]` dtype, since the
    numpy datetime returned by datashape isn't supported by pipeline.

    Parameters
    ----------
    type_: datashape.coretypes.Type
        The datashape type.

    Returns
    -------
    type_ np.dtype
        The numpy dtype.

    """
    if isinstance(type_, Option):
        type_ = type_.ty
    if isinstance(type_, DateTime):
        return np.dtype('datetime64[ns]')
    else:
        return type_.to_numpy_dtype()


@memoize
def new_dataset(expr, deltas, missing_values):
    """
    Creates or returns a dataset from a pair of blaze expressions.

    Parameters
    ----------
    expr : Expr
        The blaze expression representing the first known values.
    deltas : Expr
        The blaze expression representing the deltas to the data.
    missing_values : frozenset((name, value) pairs
        Association pairs column name and missing_value for that column.

        This needs to be a frozenset rather than a dict or tuple of tuples
        because we want a collection that's unordered but still hashable.

    Returns
    -------
    ds : type
        A new dataset type.

    Notes
    -----
    This function is memoized. repeated calls with the same inputs will return
    the same type.
    """
    missing_values = dict(missing_values)
    class_dict = {'ndim': 2 if SID_FIELD_NAME in expr.fields else 1}
    for name, type_ in expr.dshape.measure.fields:
        # Don't generate a column for sid or timestamp, since they're
        # implicitly the labels if the arrays that will be passed to pipeline
        # Terms.
        if name in (SID_FIELD_NAME, TS_FIELD_NAME):
            continue
        type_ = datashape_type_to_numpy(type_)
        if can_represent_dtype(type_):
            col = Column(
                type_,
                missing_values.get(name, NotSpecified),
            )
        else:
            col = NonPipelineField(name, type_)
        class_dict[name] = col

    name = expr._name
    if name is None:
        name = next(_new_names)

    # unicode is a name error in py3 but the branch is only hit
    # when we are in python 2.
    if PY2 and isinstance(name, unicode):  # pragma: no cover # noqa
        name = name.encode('utf-8')

    return type(name, (DataSet,), class_dict)


def _check_resources(name, expr, resources):
    """Validate that the expression and resources passed match up.

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
        If the resources do not match for an expression.
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


class NoMetaDataWarning(UserWarning):
    """Warning used to signal that no deltas or checkpoints could be found and
    none were provided.

    Parameters
    ----------
    expr : Expr
        The expression that was searched.
    field : {'deltas',  'checkpoints'}
        The field that was looked up.
    """
    def __init__(self, expr, field):
        self._expr = expr
        self._field = field

    def __str__(self):
        return 'No %s could be inferred from expr: %s' % (
            self._field,
            self._expr,
        )


no_metadata_rules = frozenset({'warn', 'raise', 'ignore'})


def _get_metadata(field, expr, metadata_expr, no_metadata_rule):
    """Find the correct metadata expression for the expression.

    Parameters
    ----------
    field : {'deltas', 'checkpoints'}
        The kind of metadata expr to lookup.
    expr : Expr
        The baseline expression.
    metadata_expr : Expr, 'auto', or None
        The metadata argument. If this is 'auto', then the metadata table will
        be searched for by walking up the expression tree. If this cannot be
        reflected, then an action will be taken based on the
        ``no_metadata_rule``.
    no_metadata_rule : {'warn', 'raise', 'ignore'}
        How to handle the case where the metadata_expr='auto' but no expr
        could be found.

    Returns
    -------
    metadata : Expr or None
        The deltas or metadata table to use.
    """
    if isinstance(metadata_expr, bz.Expr) or metadata_expr is None:
        return metadata_expr

    try:
        return expr._child['_'.join(((expr._name or ''), field))]
    except (ValueError, AttributeError):
        if no_metadata_rule == 'raise':
            raise ValueError(
                "no %s table could be reflected for %s" % (field, expr)
            )
        elif no_metadata_rule == 'warn':
            warnings.warn(NoMetaDataWarning(expr, field), stacklevel=4)
    return None


def _ad_as_ts(expr):
    """Duplicate the asof_date column as the timestamp column.

    Parameters
    ----------
    expr : Expr or None
        The expression to change the columns of.

    Returns
    -------
    transformed : Expr or None
        The transformed expression or None if ``expr`` is None.
    """
    return (
        None
        if expr is None else
        bz.transform(expr, **{TS_FIELD_NAME: expr[AD_FIELD_NAME]})
    )


def _ensure_timestamp_field(dataset_expr, deltas, checkpoints):
    """Verify that the baseline and deltas expressions have a timestamp field.

    If there is not a ``TS_FIELD_NAME`` on either of the expressions, it will
    be copied from the ``AD_FIELD_NAME``. If one is provided, then we will
    verify that it is the correct dshape.

    Parameters
    ----------
    dataset_expr : Expr
        The baseline expression.
    deltas : Expr or None
        The deltas expression if any was provided.
    checkpoints : Expr or None
        The checkpoints expression if any was provided.

    Returns
    -------
    dataset_expr, deltas : Expr
        The new baseline and deltas expressions to use.
    """
    measure = dataset_expr.dshape.measure
    if TS_FIELD_NAME not in measure.names:
        dataset_expr = bz.transform(
            dataset_expr,
            **{TS_FIELD_NAME: dataset_expr[AD_FIELD_NAME]}
        )
        deltas = _ad_as_ts(deltas)
        checkpoints = _ad_as_ts(checkpoints)
    else:
        _check_datetime_field(TS_FIELD_NAME, measure)

    return dataset_expr, deltas, checkpoints


@expect_element(
    no_deltas_rule=no_metadata_rules,
    no_checkpoints_rule=no_metadata_rules,
)
def from_blaze(expr,
               deltas='auto',
               checkpoints='auto',
               loader=None,
               resources=None,
               odo_kwargs=None,
               missing_values=None,
               no_deltas_rule='warn',
               no_checkpoints_rule='warn'):
    """Create a Pipeline API object from a blaze expression.

    Parameters
    ----------
    expr : Expr
        The blaze expression to use.
    deltas : Expr, 'auto' or None, optional
        The expression to use for the point in time adjustments.
        If the string 'auto' is passed, a deltas expr will be looked up
        by stepping up the expression tree and looking for another field
        with the name of ``expr._name`` + '_deltas'. If None is passed, no
        deltas will be used.
    deltas : Expr, 'auto' or None, optional
        The expression to use for the forward fill checkpoints.
        If the string 'auto' is passed, a checkpoints expr will be looked up
        by stepping up the expression tree and looking for another field
        with the name of ``expr._name`` + '_checkpoints'. If None is passed,
        no checkpoints will be used.
    loader : BlazeLoader, optional
        The blaze loader to attach this pipeline dataset to. If None is passed,
        the global blaze loader is used.
    resources : dict or any, optional
        The data to execute the blaze expressions against. This is used as the
        scope for ``bz.compute``.
    odo_kwargs : dict, optional
        The keyword arguments to pass to odo when evaluating the expressions.
    missing_values : dict[str -> any], optional
        A dict mapping column names to missing values for those columns.
        Missing values are required for integral columns.
    no_deltas_rule : {'warn', 'raise', 'ignore'}, optional
        What should happen if ``deltas='auto'`` but no deltas can be found.
        'warn' says to raise a warning but continue.
        'raise' says to raise an exception if no deltas can be found.
        'ignore' says take no action and proceed with no deltas.
    no_checkpoints_rule : {'warn', 'raise', 'ignore'}, optional
        What should happen if ``checkpoints='auto'`` but no checkpoints can be
        found. 'warn' says to raise a warning but continue.
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
    if 'auto' in {deltas, checkpoints}:
        invalid_nodes = tuple(filter(is_invalid_deltas_node, expr._subterms()))
        if invalid_nodes:
            raise TypeError(
                'expression with auto %s may only contain (%s) nodes,'
                " found: %s" % (
                    ' or '.join(
                        ['deltas'] if deltas is not None else [] +
                        ['checkpoints'] if checkpoints is not None else [],
                    ),
                    ', '.join(map(get__name__, valid_deltas_node_types)),
                    ', '.join(
                        set(map(compose(get__name__, type), invalid_nodes)),
                    ),
                ),
            )
    deltas = _get_metadata(
        'deltas',
        expr,
        deltas,
        no_deltas_rule,
    )
    checkpoints = _get_metadata(
        'checkpoints',
        expr,
        checkpoints,
        no_checkpoints_rule,
    )

    # Check if this is a single column out of a dataset.
    if bz.ndim(expr) != 1:
        raise TypeError(
            'expression was not tabular or array-like,'
            ' %s dimensions: %d' % (
                'too many' if bz.ndim(expr) > 1 else 'not enough',
                bz.ndim(expr),
            ),
        )

    single_column = None
    if isscalar(expr.dshape.measure):
        # This is a single column. Record which column we are to return
        # but create the entire dataset.
        single_column = rename = expr._name
        field_hit = False
        if not isinstance(expr, traversable_nodes):
            raise TypeError(
                "expression '%s' was array-like but not a simple field of"
                " some larger table" % str(expr),
            )
        while isinstance(expr, traversable_nodes):
            if isinstance(expr, bz.expr.Field):
                if not field_hit:
                    field_hit = True
                else:
                    break
            rename = expr._name
            expr = expr._child
        dataset_expr = expr.relabel({rename: single_column})
    else:
        dataset_expr = expr

    measure = dataset_expr.dshape.measure
    if not isrecord(measure) or AD_FIELD_NAME not in measure.names:
        raise TypeError(
            "The dataset must be a collection of records with at least an"
            " '{ad}' field. Fields provided: '{fields}'\nhint: maybe you need"
            " to use `relabel` to change your field names".format(
                ad=AD_FIELD_NAME,
                fields=measure,
            ),
        )
    _check_datetime_field(AD_FIELD_NAME, measure)
    dataset_expr, deltas, checkpoints = _ensure_timestamp_field(
        dataset_expr,
        deltas,
        checkpoints,
    )

    if deltas is not None and (sorted(deltas.dshape.measure.fields) !=
                               sorted(measure.fields)):
        raise TypeError(
            'baseline measure != deltas measure:\n%s != %s' % (
                measure,
                deltas.dshape.measure,
            ),
        )
    if (checkpoints is not None and
        (sorted(checkpoints.dshape.measure.fields) !=
         sorted(measure.fields))):
        raise TypeError(
            'baseline measure != checkpoints measure:\n%s != %s' % (
                measure,
                checkpoints.dshape.measure,
            ),
        )

    # Ensure that we have a data resource to execute the query against.
    _check_resources('expr', dataset_expr, resources)
    _check_resources('deltas', deltas, resources)
    _check_resources('checkpoints', checkpoints, resources)

    # Create or retrieve the Pipeline API dataset.
    if missing_values is None:
        missing_values = {}
    ds = new_dataset(dataset_expr, deltas, frozenset(missing_values.items()))

    # Register our new dataset with the loader.
    (loader if loader is not None else global_loader)[ds] = ExprData(
        bind_expression_to_resources(dataset_expr, resources),
        bind_expression_to_resources(deltas, resources)
        if deltas is not None else
        None,
        bind_expression_to_resources(checkpoints, resources)
        if checkpoints is not None else
        None,
        odo_kwargs=odo_kwargs,
    )
    if single_column is not None:
        # We were passed a single column, extract and return it.
        return getattr(ds, single_column)
    return ds


getdataset = op.attrgetter('dataset')
getname = op.attrgetter('name')


def overwrite_novel_deltas(baseline, deltas, dates):
    """overwrite any deltas into the baseline set that would have changed our
    most recently known value.

    Parameters
    ----------
    baseline : pd.DataFrame
        The first known values.
    deltas : pd.DataFrame
        Overwrites to the baseline data.
    dates : pd.DatetimeIndex
        The dates requested by the loader.

    Returns
    -------
    non_novel_deltas : pd.DataFrame
        The deltas that do not represent a baseline value.
    """
    get_indexes = dates.searchsorted
    novel_idx = (
        get_indexes(deltas[TS_FIELD_NAME].values, 'right') -
        get_indexes(deltas[AD_FIELD_NAME].values, 'left')
    ) <= 1
    novel_deltas = deltas.loc[novel_idx]
    non_novel_deltas = deltas.loc[~novel_idx]
    cat = pd.concat(
        (baseline, novel_deltas),
        ignore_index=True,
        copy=False,
    )
    sort_values(cat, TS_FIELD_NAME, inplace=True)
    return cat, non_novel_deltas


def overwrite_from_dates(asof, dense_dates, sparse_dates, asset_idx, value):
    """Construct a `Float64Overwrite` with the correct
    start and end date based on the asof date of the delta,
    the dense_dates, and the dense_dates.

    Parameters
    ----------
    asof : datetime
        The asof date of the delta.
    dense_dates : pd.DatetimeIndex
        The dates requested by the loader.
    sparse_dates : pd.DatetimeIndex
        The dates that appeared in the dataset.
    asset_idx : tuple of int
        The index of the asset in the block. If this is a tuple, then this
        is treated as the first and last index to use.
    value : np.float64
        The value to overwrite with.

    Returns
    -------
    overwrite : Float64Overwrite
        The overwrite that will apply the new value to the data.

    Notes
    -----
    This is forward-filling all dense dates that are between the asof_date date
    and the next sparse date after the asof_date.

    For example:
    let ``asof = pd.Timestamp('2014-01-02')``,
        ``dense_dates = pd.date_range('2014-01-01', '2014-01-05')``
        ``sparse_dates = pd.to_datetime(['2014-01', '2014-02', '2014-04'])``

    Then the overwrite will apply to indexes: 1, 2, 3, 4
    """
    if asof is pd.NaT:
        # Not an actual delta.
        # This happens due to the groupby we do on the deltas.
        return

    first_row = dense_dates.searchsorted(asof)
    next_idx = sparse_dates.searchsorted(asof.asm8, 'right')
    if next_idx == len(sparse_dates):
        # There is no next date in the sparse, this overwrite should apply
        # through the end of the dense dates.
        last_row = len(dense_dates) - 1
    else:
        # There is a next date in sparse dates. This means that the overwrite
        # should only apply until the index of this date in the dense dates.
        last_row = dense_dates.searchsorted(sparse_dates[next_idx]) - 1

    if first_row > last_row:
        return

    first, last = asset_idx
    yield Float64Overwrite(first_row, last_row, first, last, value)


def adjustments_from_deltas_no_sids(dense_dates,
                                    sparse_dates,
                                    column_idx,
                                    column_name,
                                    asset_idx,
                                    deltas):
    """Collect all the adjustments that occur in a dataset that does not
    have a sid column.

    Parameters
    ----------
    dense_dates : pd.DatetimeIndex
        The dates requested by the loader.
    sparse_dates : pd.DatetimeIndex
        The dates that were in the raw data.
    column_idx : int
        The index of the column in the dataset.
    column_name : str
        The name of the column to compute deltas for.
    asset_idx : pd.Series[int -> int]
        The mapping of sids to their index in the output.
    deltas : pd.DataFrame
        The overwrites that should be applied to the dataset.

    Returns
    -------
    adjustments : dict[idx -> Float64Overwrite]
        The adjustments dictionary to feed to the adjusted array.
    """
    ad_series = deltas[AD_FIELD_NAME]
    idx = 0, 0
    return {
        dense_dates.get_loc(kd): overwrite_from_dates(
            ad_series.loc[kd],
            dense_dates,
            sparse_dates,
            idx,
            v,
        ) for kd, v in deltas[column_name].iteritems()
    }


def adjustments_from_deltas_with_sids(dense_dates,
                                      sparse_dates,
                                      column_idx,
                                      column_name,
                                      asset_idx,
                                      deltas):
    """Collect all the adjustments that occur in a dataset that has a sid
    column.

    Parameters
    ----------
    dense_dates : pd.DatetimeIndex
        The dates requested by the loader.
    sparse_dates : pd.DatetimeIndex
        The dates that were in the raw data.
    column_idx : int
        The index of the column in the dataset.
    column_name : str
        The name of the column to compute deltas for.
    asset_idx : pd.Series[int -> int]
        The mapping of sids to their index in the output.
    deltas : pd.DataFrame
        The overwrites that should be applied to the dataset.

    Returns
    -------
    adjustments : dict[idx -> Float64Overwrite]
        The adjustments dictionary to feed to the adjusted array.
    """
    ad_series = deltas[AD_FIELD_NAME]
    adjustments = defaultdict(list)
    for sid, per_sid in deltas[column_name].iteritems():
        idx = asset_idx[sid]
        for kd, v in per_sid.iteritems():
            adjustments[dense_dates.searchsorted(kd)].extend(
                overwrite_from_dates(
                    ad_series.loc[kd, sid],
                    dense_dates,
                    sparse_dates,
                    (idx, idx),
                    v,
                ),
            )
    return dict(adjustments)  # no subclasses of dict


class BlazeLoader(dict):
    """A PipelineLoader for datasets constructed with ``from_blaze``.

    Parameters
    ----------
    dsmap : mapping, optional
        An initial mapping of datasets to ``ExprData`` objects.
        NOTE: Further mutations to this map will not be reflected by this
        object.
    data_query_time : time, optional
        The time to use for the data query cutoff.
    data_query_tz : tzinfo or str, optional
        The timezeone to use for the data query cutoff.
    pool : Pool, optional
        The pool to use to run blaze queries concurrently. This object must
        support ``imap_unordered``, ``apply`` and ``apply_async`` methods.

    Attributes
    ----------
    pool : Pool
        The pool to use to run blaze queries concurrently. This object must
        support ``imap_unordered``, ``apply`` and ``apply_async`` methods.
        It is possible to change the pool after the loader has been
        constructed. This allows us to set a new pool for the ``global_loader``
        like: ``global_loader.pool = multiprocessing.Pool(4)``.

    See Also
    --------
    :class:`zipline.utils.pool.SequentialPool`
    :class:`multiprocessing.Pool`
    """
    @preprocess(data_query_tz=optionally(ensure_timezone))
    def __init__(self,
                 dsmap=None,
                 data_query_time=None,
                 data_query_tz=None,
                 pool=SequentialPool()):
        self.update(dsmap or {})
        check_data_query_args(data_query_time, data_query_tz)
        self._data_query_time = data_query_time
        self._data_query_tz = data_query_tz

        # explicitly public
        self.pool = pool

    @classmethod
    @memoize(cache=WeakKeyDictionary())
    def global_instance(cls):
        return cls()

    def __hash__(self):
        return id(self)

    def __call__(self, column):
        if column.dataset in self:
            return self
        raise KeyError(column)

    def __repr__(self):
        return '<%s: %s>' % (
            type(self).__name__,
            super(BlazeLoader, self).__repr__(),
        )

    def load_adjusted_array(self, columns, dates, assets, mask):
        return merge(
            self.pool.imap_unordered(
                partial(self._load_dataset, dates, assets, mask),
                itervalues(groupby(getdataset, columns)),
            ),
        )

    def _load_dataset(self, dates, assets, mask, columns):
        try:
            (dataset,) = set(map(getdataset, columns))
        except ValueError:
            raise AssertionError('all columns must come from the same dataset')

        expr, deltas, checkpoints, odo_kwargs = self[dataset]
        have_sids = (dataset.ndim == 2)
        asset_idx = pd.Series(index=assets, data=np.arange(len(assets)))
        assets = list(map(int, assets))  # coerce from numpy.int64
        added_query_fields = [AD_FIELD_NAME, TS_FIELD_NAME] + (
            [SID_FIELD_NAME] if have_sids else []
        )
        colnames = added_query_fields + list(map(getname, columns))

        data_query_time = self._data_query_time
        data_query_tz = self._data_query_tz
        lower_dt, upper_dt = normalize_data_query_bounds(
            dates[0],
            dates[-1],
            data_query_time,
            data_query_tz,
        )

        def collect_expr(e, lower):
            """Materialize the expression as a dataframe.

            Parameters
            ----------
            e : Expr
                The baseline or deltas expression.
            lower : datetime
                The lower time bound to query.

            Returns
            -------
            result : pd.DataFrame
                The resulting dataframe.

            Notes
            -----
            This can return more data than needed. The in memory reindex will
            handle this.
            """
            predicate = e[TS_FIELD_NAME] <= upper_dt
            if lower is not None:
                predicate &= e[TS_FIELD_NAME] >= lower

            return odo(e[predicate][colnames], pd.DataFrame, **odo_kwargs)

        if checkpoints is not None:
            ts = checkpoints[TS_FIELD_NAME]
            checkpoints_ts = odo(ts[ts <= lower_dt].max(), pd.Timestamp)
            if pd.isnull(checkpoints_ts):
                materialized_checkpoints = pd.DataFrame(columns=colnames)
                lower = None
            else:
                materialized_checkpoints = odo(
                    checkpoints[ts == checkpoints_ts][colnames],
                    pd.DataFrame,
                    **odo_kwargs
                )
                lower = checkpoints_ts
        else:
            materialized_checkpoints = pd.DataFrame(columns=colnames)
            lower = None

        materialized_expr = self.pool.apply_async(collect_expr, (expr, lower))
        materialized_deltas = (
            self.pool.apply(collect_expr, (deltas, lower))
            if deltas is not None else
            pd.DataFrame(columns=colnames)
        )

        if materialized_checkpoints is not None:
            materialized_expr = pd.concat(
                (
                    materialized_checkpoints,
                    materialized_expr.get(),
                ),
                ignore_index=True,
                copy=False,
            )

        # It's not guaranteed that assets returned by the engine will contain
        # all sids from the deltas table; filter out such mismatches here.
        if not materialized_deltas.empty and have_sids:
            materialized_deltas = materialized_deltas[
                materialized_deltas[SID_FIELD_NAME].isin(assets)
            ]

        if data_query_time is not None:
            for m in (materialized_expr, materialized_deltas):
                m.loc[:, TS_FIELD_NAME] = m.loc[
                    :, TS_FIELD_NAME
                ].astype('datetime64[ns]')

                normalize_timestamp_to_query_time(
                    m,
                    data_query_time,
                    data_query_tz,
                    inplace=True,
                    ts_field=TS_FIELD_NAME,
                )

        # Inline the deltas that changed our most recently known value.
        # Also, we reindex by the dates to create a dense representation of
        # the data.
        sparse_output, non_novel_deltas = overwrite_novel_deltas(
            materialized_expr,
            materialized_deltas,
            dates,
        )
        sparse_output.drop(AD_FIELD_NAME, axis=1, inplace=True)

        sparse_deltas = last_in_date_group(non_novel_deltas,
                                           dates,
                                           assets,
                                           reindex=False)
        dense_output = last_in_date_group(sparse_output,
                                          dates,
                                          assets,
                                          reindex=True)
        ffill_across_cols(dense_output, columns)
        if have_sids:
            adjustments_from_deltas = adjustments_from_deltas_with_sids
            column_view = identity
        else:
            # If we do not have sids, use the column view to make a single
            # column vector which is unassociated with any assets.
            column_view = op.itemgetter(np.s_[:, np.newaxis])

            adjustments_from_deltas = adjustments_from_deltas_no_sids
            mask = np.full(
                shape=(len(mask), 1), fill_value=True, dtype=bool_dtype,
            )

        return {
            column: AdjustedArray(
                column_view(
                    dense_output[column.name].values.astype(column.dtype),
                ),
                mask,
                adjustments_from_deltas(
                    dates,
                    sparse_output[TS_FIELD_NAME].values,
                    column_idx,
                    column.name,
                    asset_idx,
                    sparse_deltas,
                ),
                column.missing_value,
            )
            for column_idx, column in enumerate(columns)
        }

global_loader = BlazeLoader.global_instance()


def bind_expression_to_resources(expr, resources):
    """
    Bind a Blaze expression to resources.

    Parameters
    ----------
    expr : bz.Expr
        The expression to which we want to bind resources.
    resources : dict[bz.Symbol -> any]
        Mapping from the loadable terms of ``expr`` to actual data resources.

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
        k: bz.data(v, dshape=k.dshape) for k, v in iteritems(resources)
    })


def ffill_query_in_range(expr,
                         lower,
                         upper,
                         odo_kwargs=None,
                         ts_field=TS_FIELD_NAME,
                         sid_field=SID_FIELD_NAME):
    """Query a blaze expression in a given time range properly forward filling
    from values that fall before the lower date.

    Parameters
    ----------
    expr : Expr
        Bound blaze expression.
    lower : datetime
        The lower date to query for.
    upper : datetime
        The upper date to query for.
    odo_kwargs : dict, optional
        The extra keyword arguments to pass to ``odo``.
    ts_field : str, optional
        The name of the timestamp field in the given blaze expression.
    sid_field : str, optional
        The name of the sid field in the given blaze expression.

    Returns
    -------
    raw : pd.DataFrame
        A strict dataframe for the data in the given date range. This may
        start before the requested start date if a value is needed to ffill.
    """
    odo_kwargs = odo_kwargs or {}
    filtered = expr[expr[ts_field] <= lower]
    computed_lower = odo(
        bz.by(
            filtered[sid_field],
            timestamp=filtered[ts_field].max(),
        ).timestamp.min(),
        pd.Timestamp,
        **odo_kwargs
    )
    if pd.isnull(computed_lower):
        # If there is no lower date, just query for data in the date
        # range. It must all be null anyways.
        computed_lower = lower

    raw = odo(
        expr[
            (expr[ts_field] >= computed_lower) &
            (expr[ts_field] <= upper)
        ],
        pd.DataFrame,
        **odo_kwargs
    )
    raw.loc[:, ts_field] = raw.loc[:, ts_field].astype('datetime64[ns]')
    return raw
