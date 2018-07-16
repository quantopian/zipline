"""
dataset.py
"""
from functools import total_ordering
from six import (
    iteritems,
    with_metaclass,
)

from toolz import first

from zipline.pipeline.classifiers import Classifier, Latest as LatestClassifier
from zipline.pipeline.factors import Factor, Latest as LatestFactor
from zipline.pipeline.filters import Filter, Latest as LatestFilter
from zipline.pipeline.sentinels import NotSpecified
from zipline.pipeline.term import (
    AssetExists,
    LoadableTerm,
    validate_dtype,
)
from zipline.utils.input_validation import ensure_dtype
from zipline.utils.numpy_utils import NoDefaultMissingValue
from zipline.utils.preprocess import preprocess


class Column(object):
    """
    An abstract column of data, not yet associated with a dataset.
    """
    @preprocess(dtype=ensure_dtype)
    def __init__(self,
                 dtype,
                 missing_value=NotSpecified,
                 doc=None,
                 metadata=None):
        self.dtype = dtype
        self.missing_value = missing_value
        self.doc = doc
        self.metadata = metadata.copy() if metadata is not None else {}

    def bind(self, name):
        """
        Bind a `Column` object to its name.
        """
        return _BoundColumnDescr(
            dtype=self.dtype,
            missing_value=self.missing_value,
            name=name,
            doc=self.doc,
            metadata=self.metadata,
        )


class _BoundColumnDescr(object):
    """
    Intermediate class that sits on `DataSet` objects and returns memoized
    `BoundColumn` objects when requested.

    This exists so that subclasses of DataSets don't share columns with their
    parent classes.
    """
    def __init__(self, dtype, missing_value, name, doc, metadata):
        # Validating and calculating default missing values here guarantees
        # that we fail quickly if the user passes an unsupporte dtype or fails
        # to provide a missing value for a dtype that requires one
        # (e.g. int64), but still enables us to provide an error message that
        # points to the name of the failing column.
        try:
            self.dtype, self.missing_value = validate_dtype(
                termname="Column(name={name!r})".format(name=name),
                dtype=dtype,
                missing_value=missing_value,
            )
        except NoDefaultMissingValue:
            # Re-raise with a more specific message.
            raise NoDefaultMissingValue(
                "Failed to create Column with name {name!r} and"
                " dtype {dtype} because no missing_value was provided\n\n"
                "Columns with dtype {dtype} require a missing_value.\n"
                "Please pass missing_value to Column() or use a different"
                " dtype.".format(dtype=dtype, name=name)
            )
        self.name = name
        self.doc = doc
        self.metadata = metadata

    def __get__(self, instance, owner):
        """
        Produce a concrete BoundColumn object when accessed.

        We don't bind to datasets at class creation time so that subclasses of
        DataSets produce different BoundColumns.
        """
        return BoundColumn(
            dtype=self.dtype,
            missing_value=self.missing_value,
            dataset=owner,
            name=self.name,
            doc=self.doc,
            metadata=self.metadata,
        )


class BoundColumn(LoadableTerm):
    """
    A column of data that's been concretely bound to a particular dataset.

    Instances of this class are dynamically created upon access to attributes
    of DataSets (for example, USEquityPricing.close is an instance of this
    class).

    Attributes
    ----------
    dtype : numpy.dtype
        The dtype of data produced when this column is loaded.
    latest : zipline.pipeline.data.Factor or zipline.pipeline.data.Filter
        A Filter, Factor, or Classifier computing the most recently known value
        of this column on each date.

        Produces a Filter if self.dtype == ``np.bool_``.
        Produces a Classifier if self.dtype == ``np.int64``
        Otherwise produces a Factor.
    dataset : zipline.pipeline.data.DataSet
        The dataset to which this column is bound.
    name : str
        The name of this column.
    metadata : dict
        Extra metadata associated with this column.
    """
    mask = AssetExists()
    window_safe = True

    def __new__(cls, dtype, missing_value, dataset, name, doc, metadata):
        return super(BoundColumn, cls).__new__(
            cls,
            domain=dataset.domain,
            dtype=dtype,
            missing_value=missing_value,
            dataset=dataset,
            name=name,
            ndim=dataset.ndim,
            doc=doc,
            metadata=metadata,
        )

    def _init(self, dataset, name, doc, metadata, *args, **kwargs):
        self._dataset = dataset
        self._name = name
        self.__doc__ = doc
        self._metadata = metadata
        return super(BoundColumn, self)._init(*args, **kwargs)

    @classmethod
    def _static_identity(cls, dataset, name, doc, metadata, *args, **kwargs):
        return (
            super(BoundColumn, cls)._static_identity(*args, **kwargs),
            dataset,
            name,
            doc,
            frozenset(sorted(metadata.items(), key=first)),
        )

    @property
    def dataset(self):
        """
        The dataset to which this column is bound.
        """
        return self._dataset

    @property
    def name(self):
        """
        The name of this column.
        """
        return self._name

    @property
    def metadata(self):
        """
        A copy of the metadata for this column.
        """
        return self._metadata.copy()

    @property
    def qualname(self):
        """
        The fully-qualified name of this column.

        Generated by doing '.'.join([self.dataset.__name__, self.name]).
        """
        return '.'.join([self.dataset.__name__, self.name])

    @property
    def latest(self):
        dtype = self.dtype
        if dtype in Filter.ALLOWED_DTYPES:
            Latest = LatestFilter
        elif dtype in Classifier.ALLOWED_DTYPES:
            Latest = LatestClassifier
        else:
            assert dtype in Factor.ALLOWED_DTYPES, "Unknown dtype %s." % dtype
            Latest = LatestFactor

        return Latest(
            inputs=(self,),
            dtype=dtype,
            missing_value=self.missing_value,
            ndim=self.ndim,
        )

    def __repr__(self):
        return "{qualname}::{dtype}".format(
            qualname=self.qualname,
            dtype=self.dtype.name,
        )

    def graph_repr(self):
        """Short repr to use when rendering Pipeline graphs."""
        return "BoundColumn:\l  Dataset: {}\l  Column: {}\l".format(
            self.dataset.__name__,
            self.name
        )

    def recursive_repr(self):
        """Short repr used to render in recursive contexts."""
        return self.qualname


@total_ordering
class DataSetMeta(type):
    """
    Metaclass for DataSets

    Supplies name and dataset information to Column attributes.
    """

    def __new__(mcls, name, bases, dict_):
        newtype = super(DataSetMeta, mcls).__new__(mcls, name, bases, dict_)
        # collect all of the column names that we inherit from our parents
        column_names = set().union(
            *(getattr(base, '_column_names', ()) for base in bases)
        )
        for maybe_colname, maybe_column in iteritems(dict_):
            if isinstance(maybe_column, Column):
                # add column names defined on our class
                bound_column_descr = maybe_column.bind(maybe_colname)
                setattr(newtype, maybe_colname, bound_column_descr)
                column_names.add(maybe_colname)

        newtype._column_names = frozenset(column_names)
        return newtype

    @property
    def columns(self):
        return frozenset(
            getattr(self, colname) for colname in self._column_names
        )

    def __lt__(self, other):
        return id(self) < id(other)

    def __repr__(self):
        return '<DataSet: %r>' % self.__name__


class DataSet(with_metaclass(DataSetMeta, object)):
    """
    Base class for describing inputs to Pipeline expressions.

    A DataSet is a collection of :class:`zipline.pipeline.data.Column` that
    describes a collection of logically-related inputs to the Pipeline API.

    To create a new Pipeline dataset, subclass from this class and create
    columns at class scope for each attribute of your dataset. Each column
    requires a dtype that describes the type of data that should be produced by
    a loader for the dataset. Integer columns must also provide a
    ``missing_value`` to be used when no value is available for a given
    asset/date combination.

    Examples
    --------
    The built-in USEquityPricing dataset is defined as follows::

        class EquityPricing(DataSet):
            open = Column(float)
            high = Column(float)
            low = Column(float)
            close = Column(float)
            volume = Column(float)

    Columns can have types other than float. A dataset containing assorted
    company metadata might be defined like this::

        class CompanyMetadata(DataSet):
            # Use float for semantically-numeric data, even if it's always
            # integral valued (see Notes section below). The default missing
            # value for floats is NaN.
            shares_outstanding = Column(float)

            # Use object-dtype for string columns. The default missing value
            # for object-dtype columns is None.
            ticker = Column(object)

            # Use integers for integer-valued categorical data like sector or
            # industry codes. Integer-dtype columns require an explicit missing
            # value.
            sector_code = Column(int, missing_value=-1)

            # The default missing value for bool-dtype columns is False.
            is_primary_share = Column(bool)

    Notes
    -----
    Because numpy has no native support for integers with missing values, users
    are strongly encouraged to use floats for any data that's semantically
    numeric. Doing so enables the use of `NaN` as a natural missing value,
    which has useful propagation semantics.
    """
    domain = None
    ndim = 2
