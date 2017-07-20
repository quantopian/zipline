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
from collections import namedtuple
from functools import partial
from itertools import count
import warnings
from weakref import WeakKeyDictionary

import blaze as bz
from datashape import (
    Date,
    DateTime,
    Option,
    String,
    isrecord,
    isscalar,
    integral,
)
import numpy as np
from odo import odo
import pandas as pd
from six import with_metaclass, PY2, itervalues, iteritems
from toolz import (
    complement,
    compose,
    first,
    flip,
    groupby,
    memoize,
    merge,
)
import toolz.curried.operator as op
from toolz.curried.operator import getitem

from zipline.pipeline.common import (
    AD_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME
)
from zipline.pipeline.data.dataset import DataSet, Column
from zipline.pipeline.loaders.utils import (
    check_data_query_args,
    normalize_data_query_bounds,
)
from zipline.pipeline.sentinels import NotSpecified
from zipline.lib.adjusted_array import can_represent_dtype
from zipline.utils.input_validation import (
    expect_element,
    ensure_timezone,
    optionally,
)
from zipline.utils.pool import SequentialPool
from zipline.utils.preprocess import preprocess
from ._core import (
    adjusted_arrays_from_rows_with_assets,
    adjusted_arrays_from_rows_without_assets,
)


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
    if isinstance(type_, String):
        return np.dtype(object)
    if type_ in integral:
        return np.dtype('int64')
    else:
        return type_.to_numpy_dtype()


@memoize
def new_dataset(expr, missing_values):
    """
    Creates or returns a dataset from a pair of blaze expressions.

    Parameters
    ----------
    expr : Expr
        The blaze expression representing the first known values.
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
               no_checkpoints_rule='warn',
               apply_deltas_adjustments=True,):
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
    checkpoints : Expr, 'auto' or None, optional
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
    apply_deltas_adjustments : bool, optional
        Whether or not deltas adjustments should be applied for this dataset.
        True by default because not applying deltas adjustments is an exception
        rather than the rule.

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
    ds = new_dataset(dataset_expr, frozenset(missing_values.items()))

    # Register our new dataset with the loader.
    (loader if loader is not None else global_loader).register_dataset(
        ds,
        bind_expression_to_resources(dataset_expr, resources),
        bind_expression_to_resources(deltas, resources)
        if deltas is not None else
        None,
        bind_expression_to_resources(checkpoints, resources)
        if checkpoints is not None else
        None,
        odo_kwargs=odo_kwargs,
        apply_deltas_adjustments=apply_deltas_adjustments
    )
    if single_column is not None:
        # We were passed a single column, extract and return it.
        return getattr(ds, single_column)
    return ds


getdataset = op.attrgetter('dataset')
getname = op.attrgetter('name')

_expr_data_base = namedtuple(
    'ExprData', 'expr deltas checkpoints odo_kwargs apply_deltas_adjustments'
)


class ExprData(_expr_data_base):
    """A pair of expressions and data resources. The expressions will be
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
    apply_deltas_adjustments : bool, optional
        Whether or not deltas adjustments should be applied to the baseline
        values. If False, only novel deltas will be applied.
    """
    def __new__(cls,
                expr,
                deltas=None,
                checkpoints=None,
                odo_kwargs=None,
                apply_deltas_adjustments=True):
        return super(ExprData, cls).__new__(
            cls,
            expr,
            deltas,
            checkpoints,
            odo_kwargs or {},
            apply_deltas_adjustments,
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
            self.apply_deltas_adjustments,
        ))

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class BlazeLoader(object):
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
        check_data_query_args(data_query_time, data_query_tz)
        self._data_query_time = data_query_time
        self._data_query_tz = data_query_tz

        # explicitly public
        self.pool = pool

        self._table_expressions = (dsmap or {}).copy()

    @classmethod
    @memoize(cache=WeakKeyDictionary())
    def global_instance(cls):
        return cls()

    def __hash__(self):
        return id(self)

    def __contains__(self, column):
        return column in self._table_expressions

    def __getitem__(self, column):
        return self._table_expressions[column]

    def __iter__(self):
        return iter(self._table_expressions)

    def __len__(self):
        return len(self._table_expressions)

    def __call__(self, column):
        if column in self:
            return self
        raise KeyError(column)

    def register_dataset(self,
                         dataset,
                         expr,
                         deltas=None,
                         checkpoints=None,
                         odo_kwargs=None,
                         apply_deltas_adjustments=True):
        """Explicitly map a datset to a collection of blaze expressions.

        Parameters
        ----------
        dataset : DataSet
            The pipeline dataset to map to the given expressions.
        expr : Expr
            The baseline values.
        deltas : Expr, optional
            The deltas for the data.
        checkpoints : Expr, optional
            The forward fill checkpoints for the data.
        odo_kwargs : dict, optional
            The keyword arguments to forward to the odo calls internally.
        apply_deltas_adjustments : bool, optional
            Whether or not deltas adjustments should be applied to the baseline
            values. If False, only novel deltas will be applied.

        See Also
        --------
        :func:`zipline.pipeline.loaders.blaze.from_blaze`
        """
        expr_data = ExprData(
            expr,
            deltas,
            checkpoints,
            odo_kwargs,
            apply_deltas_adjustments,
        )
        for column in dataset.columns:
            self._table_expressions[column] = expr_data

    def register_column(self,
                        column,
                        expr,
                        deltas=None,
                        checkpoints=None,
                        odo_kwargs=None,
                        apply_deltas_adjustments=True):
        """Explicitly map a single bound column to a collection of blaze
        expressions. The expressions need to have ``timestamp`` and ``as_of``
        columns.

        Parameters
        ----------
        column : BoundColumn
            The pipeline dataset to map to the given expressions.
        expr : Expr
            The baseline values.
        deltas : Expr, optional
            The deltas for the data.
        checkpoints : Expr, optional
            The forward fill checkpoints for the data.
        odo_kwargs : dict, optional
            The keyword arguments to forward to the odo calls internally.
        apply_deltas_adjustments : bool, optional
            Whether or not deltas adjustments should be applied to the baseline
            values. If False, only novel deltas will be applied.

        See Also
        --------
        :func:`zipline.pipeline.loaders.blaze.from_blaze`
        """
        self._table_expressions[column] = ExprData(
            expr,
            deltas,
            checkpoints,
            odo_kwargs,
            apply_deltas_adjustments,
        )

    def load_adjusted_array(self, columns, dates, assets, mask):
        return merge(
            self.pool.imap_unordered(
                partial(self._load_dataset, dates, assets, mask),
                itervalues(groupby(getitem(self._table_expressions), columns)),
            ),
        )

    def _load_dataset(self, dates, assets, mask, columns):
        try:
            (expr_data,) = {self._table_expressions[c] for c in columns}
        except ValueError:
            raise AssertionError(
                'all columns must share the same expression data',
            )

        (expr,
         deltas,
         checkpoints,
         odo_kwargs,
         apply_deltas_adjustments) = expr_data

        have_sids = (first(columns).dataset.ndim == 2)
        added_query_fields = {AD_FIELD_NAME, TS_FIELD_NAME} | (
            {SID_FIELD_NAME} if have_sids else set()
        )
        requested_columns = set(map(getname, columns))
        colnames = sorted(added_query_fields | requested_columns)

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

        lower, materialized_checkpoints = get_materialized_checkpoints(
            checkpoints, colnames, lower_dt, odo_kwargs
        )

        materialized_expr_deferred = self.pool.apply_async(
            collect_expr,
            (expr, lower),
        )
        materialized_deltas = (
            self.pool.apply(collect_expr, (deltas, lower))
            if deltas is not None and apply_deltas_adjustments else
            None
        )

        all_rows = pd.concat(
            filter(
                lambda df: df is not None, (
                    materialized_checkpoints,
                    materialized_expr_deferred.get(),
                    materialized_deltas,
                ),
            ),
            ignore_index=True,
            copy=False,
        )

        if data_query_time is not None:
            ts_dates = pd.DatetimeIndex([
                pd.Timestamp.combine(
                    dt.date(),
                    data_query_time,
                ).tz_localize(data_query_tz).tz_convert('utc')
                for dt in dates
            ])
        else:
            ts_dates = dates

        all_rows[TS_FIELD_NAME] = all_rows[TS_FIELD_NAME].astype(
            'datetime64[ns]',
        )
        all_rows.sort_values([TS_FIELD_NAME, AD_FIELD_NAME], inplace=True)
        # from nose.tools import set_trace;set_trace()
        if have_sids:
            return adjusted_arrays_from_rows_with_assets(
                dates,
                ts_dates,
                assets,
                mask,
                columns,
                all_rows,
            )
        else:
            return adjusted_arrays_from_rows_without_assets(
                dates,
                ts_dates,
                None,
                columns,
                all_rows,
            )


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


def get_materialized_checkpoints(checkpoints, colnames, lower_dt, odo_kwargs):
    """
    Computes a lower bound and a DataFrame checkpoints.

    Parameters
    ----------
    checkpoints : Expr
        Bound blaze expression for a checkpoints table from which to get a
        computed lower bound.
    colnames : iterable of str
        The names of the columns for which checkpoints should be computed.
    lower_dt : pd.Timestamp
        The lower date being queried for that serves as an upper bound for
        checkpoints.
    odo_kwargs : dict, optional
        The extra keyword arguments to pass to ``odo``.
    """
    if checkpoints is not None:
        ts = checkpoints[TS_FIELD_NAME]
        checkpoints_ts = odo(
            ts[ts <= lower_dt].max(),
            pd.Timestamp,
            **odo_kwargs
        )
        if pd.isnull(checkpoints_ts):
            # We don't have a checkpoint for before our start date so just
            # don't constrain the lower date.
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
        lower = None  # we don't have a good lower date constraint
    return lower, materialized_checkpoints


def ffill_query_in_range(expr,
                         lower,
                         upper,
                         checkpoints=None,
                         odo_kwargs=None,
                         ts_field=TS_FIELD_NAME):
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
    checkpoints : Expr, optional
        Bound blaze expression for a checkpoints table from which to get a
        computed lower bound.
    odo_kwargs : dict, optional
        The extra keyword arguments to pass to ``odo``.
    ts_field : str, optional
        The name of the timestamp field in the given blaze expression.

    Returns
    -------
    raw : pd.DataFrame
        A strict dataframe for the data in the given date range. This may
        start before the requested start date if a value is needed to ffill.
    """
    odo_kwargs = odo_kwargs or {}
    computed_lower, materialized_checkpoints = get_materialized_checkpoints(
        checkpoints,
        expr.fields,
        lower,
        odo_kwargs,
    )

    pred = expr[ts_field] <= upper

    if computed_lower is not None:
        # only constrain the lower date if we computed a new lower date
        pred &= expr[ts_field] >= computed_lower

    raw = pd.concat(
        (
            materialized_checkpoints,
            odo(
                expr[pred],
                pd.DataFrame,
                **odo_kwargs
            ),
        ),
        ignore_index=True,
    )
    raw.loc[:, ts_field] = raw.loc[:, ts_field].astype('datetime64[ns]')
    return raw
