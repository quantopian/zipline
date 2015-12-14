"""
dataset.py
"""
from functools import total_ordering
from six import (
    iteritems,
    with_metaclass,
)

from zipline.pipeline.term import Term, AssetExists
from zipline.utils.input_validation import ensure_dtype
from zipline.utils.preprocess import preprocess


class Column(object):
    """
    An abstract column of data, not yet associated with a dataset.
    """

    @preprocess(dtype=ensure_dtype)
    def __init__(self, dtype):
        self.dtype = dtype

    def bind(self, dataset, name):
        """
        Bind a column to a concrete dataset.
        """
        return BoundColumn(dtype=self.dtype, dataset=dataset, name=name)


class BoundColumn(Term):
    """
    A Column of data that's been concretely bound to a particular dataset.
    """
    mask = AssetExists()
    extra_input_rows = 0
    inputs = ()

    def __new__(cls, dtype, dataset, name):
        return super(BoundColumn, cls).__new__(
            cls,
            domain=dataset.domain,
            dtype=dtype,
            dataset=dataset,
            name=name,
        )

    def _init(self, dataset, name, *args, **kwargs):
        self._dataset = dataset
        self._name = name
        return super(BoundColumn, self)._init(*args, **kwargs)

    @classmethod
    def static_identity(cls, dataset, name, *args, **kwargs):
        return (
            super(BoundColumn, cls).static_identity(*args, **kwargs),
            dataset,
            name,
        )

    @property
    def dataset(self):
        return self._dataset

    @property
    def name(self):
        return self._name

    @property
    def qualname(self):
        """
        Fully qualified of this column.
        """
        return '.'.join([self.dataset.__name__, self.name])

    @property
    def latest(self):
        from zipline.pipeline.factors import Latest
        return Latest(inputs=(self,), dtype=self.dtype)

    def __repr__(self):
        return "{qualname}::{dtype}".format(
            qualname=self.qualname,
            dtype=self.dtype.name,
        )

    def short_repr(self):
        return self.qualname


@total_ordering
class DataSetMeta(type):
    """
    Metaclass for DataSets

    Supplies name and dataset information to Column attributes.
    """

    def __new__(mcls, name, bases, dict_):
        newtype = type.__new__(mcls, name, bases, dict_)
        _columns = []
        for maybe_colname, maybe_column in iteritems(dict_):
            if isinstance(maybe_column, Column):
                bound_column = maybe_column.bind(newtype, maybe_colname)
                setattr(newtype, maybe_colname, bound_column)
                _columns.append(bound_column)

        newtype._columns = frozenset(_columns)
        return newtype

    @property
    def columns(self):
        return self._columns

    def __lt__(self, other):
        return id(self) < id(other)

    def __repr__(self):
        return '<DataSet: %r>' % self.__name__


class DataSet(with_metaclass(DataSetMeta)):
    domain = None
