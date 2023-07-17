import abc
from collections import namedtuple, OrderedDict
from itertools import repeat
from textwrap import dedent
from weakref import WeakKeyDictionary

from toolz import first

from zipline.currency import Currency
from zipline.data.fx import DEFAULT_FX_RATE
from zipline.pipeline.classifiers import Classifier, Latest as LatestClassifier
from zipline.pipeline.domain import Domain, GENERIC
from zipline.pipeline.factors import Factor, Latest as LatestFactor
from zipline.pipeline.filters import Filter, Latest as LatestFilter
from zipline.pipeline.sentinels import NotSpecified, sentinel
from zipline.pipeline.term import (
    AssetExists,
    LoadableTerm,
    validate_dtype,
)
from zipline.utils.formatting import s, plural
from zipline.utils.input_validation import (
    coerce_types,
    ensure_dtype,
    expect_types,
)
from zipline.utils.numpy_utils import float64_dtype, NoDefaultMissingValue
from zipline.utils.preprocess import preprocess
from zipline.utils.string_formatting import bulleted_list

IsSpecialization = sentinel("IsSpecialization")


class Column:
    """
    An abstract column of data, not yet associated with a dataset.
    """

    @preprocess(dtype=ensure_dtype)
    def __init__(
        self,
        dtype,
        missing_value=NotSpecified,
        doc=None,
        metadata=None,
        currency_aware=False,
    ):
        if currency_aware and dtype != float64_dtype:
            raise ValueError(
                "Columns cannot be constructed with currency_aware={}, "
                "dtype={}. Currency aware columns must have a float64 dtype.".format(
                    currency_aware, dtype
                )
            )

        self.dtype = dtype
        self.missing_value = missing_value
        self.doc = doc
        self.metadata = metadata.copy() if metadata is not None else {}
        self.currency_aware = currency_aware

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
            currency_aware=self.currency_aware,
        )


class _BoundColumnDescr:
    """
    Intermediate class that sits on `DataSet` objects and returns memoized
    `BoundColumn` objects when requested.

    This exists so that subclasses of DataSets don't share columns with their
    parent classes.
    """

    def __init__(self, dtype, missing_value, name, doc, metadata, currency_aware):
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
        except NoDefaultMissingValue as exc:
            # Re-raise with a more specific message.
            raise NoDefaultMissingValue(
                "Failed to create Column with name {name!r} and"
                " dtype {dtype} because no missing_value was provided\n\n"
                "Columns with dtype {dtype} require a missing_value.\n"
                "Please pass missing_value to Column() or use a different"
                " dtype.".format(dtype=dtype, name=name)
            ) from exc
        self.name = name
        self.doc = doc
        self.metadata = metadata
        self.currency_aware = currency_aware

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
            currency_conversion=None,
            currency_aware=self.currency_aware,
        )


class BoundColumn(LoadableTerm):
    """
    A column of data that's been concretely bound to a particular dataset.

    Attributes
    ----------
    dtype : numpy.dtype
        The dtype of data produced when this column is loaded.
    latest : zipline.pipeline.LoadableTerm
        A :class:`~zipline.pipeline.Filter`, :class:`~zipline.pipeline.Factor`,
        or :class:`~zipline.pipeline.Classifier` computing the most recently
        known value of this column on each date.
        See :class:`zipline.pipeline.mixins.LatestMixin` for more details.
    dataset : zipline.pipeline.data.DataSet
        The dataset to which this column is bound.
    name : str
        The name of this column.
    metadata : dict
        Extra metadata associated with this column.
    currency_aware : bool
        Whether or not this column produces currency-denominated data.

    Notes
    -----
    Instances of this class are dynamically created upon access to attributes
    of :class:`~zipline.pipeline.data.DataSet`. For example,
    :attr:`~zipline.pipeline.data.EquityPricing.close` is an instance of this
    class. Pipeline API users should never construct instances of this
    directly.
    """

    mask = AssetExists()
    window_safe = True

    def __new__(
        cls,
        dtype,
        missing_value,
        dataset,
        name,
        doc,
        metadata,
        currency_conversion,
        currency_aware,
    ):
        if currency_aware and dtype != float64_dtype:
            raise AssertionError(
                "The {} column on dataset {} cannot be constructed with "
                "currency_aware={}, dtype={}. Currency aware columns must "
                "have a float64 dtype.".format(
                    name,
                    dataset,
                    currency_aware,
                    dtype,
                )
            )

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
            currency_conversion=currency_conversion,
            currency_aware=currency_aware,
        )

    def _init(
        self,
        dataset,
        name,
        doc,
        metadata,
        currency_conversion,
        currency_aware,
        *args,
        **kwargs,
    ):
        self._dataset = dataset
        self._name = name
        self.__doc__ = doc
        self._metadata = metadata
        self._currency_conversion = currency_conversion
        self._currency_aware = currency_aware
        return super(BoundColumn, self)._init(*args, **kwargs)

    @classmethod
    def _static_identity(
        cls,
        dataset,
        name,
        doc,
        metadata,
        currency_conversion,
        currency_aware,
        *args,
        **kwargs,
    ):
        return (
            super(BoundColumn, cls)._static_identity(*args, **kwargs),
            dataset,
            name,
            doc,
            frozenset(sorted(metadata.items(), key=first)),
            currency_conversion,
            currency_aware,
        )

    def __lt__(self, other):
        msg = "Can't compare '{}' with '{}'. (Did you mean to use '.latest'?)"
        raise TypeError(msg.format(self.qualname, other.__class__.__name__))

    __gt__ = __le__ = __ge__ = __lt__

    def _replace(self, **kwargs):
        kw = dict(
            dtype=self.dtype,
            missing_value=self.missing_value,
            dataset=self._dataset,
            name=self._name,
            doc=self.__doc__,
            metadata=self._metadata,
            currency_conversion=self._currency_conversion,
            currency_aware=self._currency_aware,
        )
        kw.update(kwargs)

        return type(self)(**kw)

    def specialize(self, domain):
        """Specialize ``self`` to a concrete domain."""
        if domain == self.domain:
            return self

        return self._replace(dataset=self._dataset.specialize(domain))

    def unspecialize(self):
        """
        Unspecialize a column to its generic form.

        This is equivalent to ``column.specialize(GENERIC)``.
        """
        return self.specialize(GENERIC)

    @coerce_types(currency=(str, Currency))
    def fx(self, currency):
        """
        Construct a currency-converted version of this column.

        Parameters
        ----------
        currency : str or zipline.currency.Currency
            Currency into which to convert this column's data.

        Returns
        -------
        column : BoundColumn
            Column producing the same data as ``self``, but currency-converted
            into ``currency``.
        """
        conversion = self._currency_conversion

        if not self._currency_aware:
            raise TypeError(
                "The .fx() method cannot be called on {} because it does not "
                "produce currency-denominated data.".format(self.qualname)
            )
        elif conversion is not None and conversion.currency == currency:
            return self

        return self._replace(
            currency_conversion=CurrencyConversion(
                currency=currency,
                field=DEFAULT_FX_RATE,
            )
        )

    @property
    def currency_conversion(self):
        """Specification for currency conversions applied for this term."""
        return self._currency_conversion

    @property
    def currency_aware(self):
        """
        Whether or not this column produces currency-denominated data.
        """
        return self._currency_aware

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
        """The fully-qualified name of this column."""
        out = ".".join([self.dataset.qualname, self.name])
        conversion = self._currency_conversion
        if conversion is not None:
            out += ".fx({!r})".format(conversion.currency.code)
        return out

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
        # Graphviz interprets `\l` as "divide label into lines, left-justified"
        return "BoundColumn:\\l  Dataset: {}\\l  Column: {}\\l".format(
            self.dataset.__name__, self.name
        )

    def recursive_repr(self):
        """Short repr used to render in recursive contexts."""
        return self.qualname


class DataSetMeta(type):
    """
    Metaclass for DataSets

    Supplies name and dataset information to Column attributes, and manages
    families of specialized dataset.
    """

    def __new__(metacls, name, bases, dict_):
        if len(bases) != 1:
            # Disallowing multiple inheritance makes it easier for us to
            # determine whether a given dataset is the root for its family of
            # specializations.
            raise TypeError("Multiple dataset inheritance is not supported.")

        # This marker is set in the class dictionary by `specialize` below.
        is_specialization = dict_.pop(IsSpecialization, False)

        newtype = super(DataSetMeta, metacls).__new__(metacls, name, bases, dict_)

        if not isinstance(newtype.domain, Domain):
            raise TypeError(
                "Expected a Domain for {}.domain, but got {} instead.".format(
                    newtype.__name__,
                    type(newtype.domain),
                )
            )

        # Collect all of the column names that we inherit from our parents.
        column_names = set().union(
            *(getattr(base, "_column_names", ()) for base in bases)
        )

        # Collect any new columns from this dataset.
        for maybe_colname, maybe_column in dict_.items():
            if isinstance(maybe_column, Column):
                # add column names defined on our class
                bound_column_descr = maybe_column.bind(maybe_colname)
                setattr(newtype, maybe_colname, bound_column_descr)
                column_names.add(maybe_colname)

        newtype._column_names = frozenset(column_names)

        if not is_specialization:
            # This is the new root of a family of specializations. Store the
            # memoized dictionary for family on this type.
            newtype._domain_specializations = WeakKeyDictionary(
                {
                    newtype.domain: newtype,
                }
            )

        return newtype

    @expect_types(domain=Domain)
    def specialize(cls, domain):
        """
        Specialize a generic DataSet to a concrete domain.

        Parameters
        ----------
        domain : zipline.pipeline.domain.Domain
            Domain to which we should generate a specialization.

        Returns
        -------
        specialized : type
            A new :class:`~zipline.pipeline.data.DataSet` subclass with the
            same columns as ``self``, but specialized to ``domain``.
        """
        # We're already the specialization to this domain, so just return self.
        if domain == cls.domain:
            return cls

        try:
            return cls._domain_specializations[domain]
        except KeyError as exc:
            if not cls._can_create_new_specialization(domain):
                # This either means we're already a specialization and trying
                # to create a new specialization, or we're the generic version
                # of a root-specialized dataset, which we don't want to create
                # new specializations of.
                raise ValueError(
                    "Can't specialize {dataset} from {current} to new domain {new}.".format(
                        dataset=cls.__name__,
                        current=cls.domain,
                        new=domain,
                    )
                ) from exc
            new_type = cls._create_specialization(domain)
            cls._domain_specializations[domain] = new_type
            return new_type

    def unspecialize(cls):
        """
        Unspecialize a dataset to its generic form.

        This is equivalent to ``dataset.specialize(GENERIC)``.
        """
        return cls.specialize(GENERIC)

    def _can_create_new_specialization(cls, domain):
        # Always allow specializing to a generic domain.
        if domain is GENERIC:
            return True
        elif "_domain_specializations" in vars(cls):
            # This branch is True if we're the root of a family.
            # Allow specialization if we're generic.
            return cls.domain is GENERIC
        else:
            # If we're not the root of a family, we can't create any new
            # specializations.
            return False

    def _create_specialization(cls, domain):
        # These are all assertions because we should have handled these cases
        # already in specialize().
        assert isinstance(domain, Domain)
        assert (
            domain not in cls._domain_specializations
        ), "Domain specializations should be memoized!"
        if domain is not GENERIC:
            assert (
                cls.domain is GENERIC
            ), "Can't specialize dataset with domain {} to domain {}.".format(
                cls.domain,
                domain,
            )

        # Create a new subclass of ``self`` with the given domain.
        # Mark that it's a specialization so that we know not to create a new
        # family for it.
        name = cls.__name__
        bases = (cls,)
        dict_ = {"domain": domain, IsSpecialization: True}
        out = type(name, bases, dict_)
        out.__module__ = cls.__module__
        return out

    @property
    def columns(cls):
        return frozenset(getattr(cls, colname) for colname in cls._column_names)

    @property
    def qualname(cls):
        if cls.domain is GENERIC:
            specialization_key = ""
        else:
            specialization_key = "<" + cls.domain.country_code + ">"

        return cls.__name__ + specialization_key

    # NOTE: We used to use `functools.total_ordering` to account for all of the
    #       other rich comparison methods, but it has issues in python 3 and
    #       this method is only used for test purposes, so for now we will just
    #       keep this in isolation. If we ever need any of the other comparison
    #       methods we will have to implement them individually.
    def __lt__(cls, other):
        return id(cls) < id(other)

    def __repr__(cls):
        return "<DataSet: %r, domain=%s>" % (cls.__name__, cls.domain)


class DataSet(object, metaclass=DataSetMeta):
    """
    Base class for Pipeline datasets.

    A :class:`DataSet` is defined by two parts:

    1. A collection of :class:`~zipline.pipeline.data.Column` objects that
       describe the queryable attributes of the dataset.

    2. A :class:`~zipline.pipeline.domain.Domain` describing the assets and
       calendar of the data represented by the :class:`DataSet`.

    To create a new Pipeline dataset, define a subclass of :class:`DataSet` and
    set one or more :class:`Column` objects as class-level attributes. Each
    column requires a ``np.dtype`` that describes the type of data that should
    be produced by a loader for the dataset. Integer columns must also provide
    a "missing value" to be used when no value is available for a given
    asset/date combination.

    By default, the domain of a dataset is the special singleton value,
    :data:`~zipline.pipeline.domain.GENERIC`, which means that they can be used
    in a Pipeline running on **any** domain.

    In some cases, it may be preferable to restrict a dataset to only allow
    support a single domain. For example, a DataSet may describe data from a
    vendor that only covers the US. To restrict a dataset to a specific domain,
    define a `domain` attribute at class scope.

    You can also define a domain-specific version of a generic DataSet by
    calling its ``specialize`` method with the domain of interest.

    Examples
    --------
    The built-in EquityPricing dataset is defined as follows::

        class EquityPricing(DataSet):
            open = Column(float)
            high = Column(float)
            low = Column(float)
            close = Column(float)
            volume = Column(float)

    The built-in USEquityPricing dataset is a specialization of
    EquityPricing. It is defined as::

        from zipline.pipeline.domain import US_EQUITIES
        USEquityPricing = EquityPricing.specialize(US_EQUITIES)

    Columns can have types other than float. A dataset containing assorted
    company metadata might be defined like this::

        class CompanyMetadata(DataSet):
            # Use float for semantically-numeric data, even if it's always
            # integral valued (see Notes section below). The default missing
            # value for floats is NaN.
            shares_outstanding = Column(float)

            # Use object for string columns. The default missing value for
            # object-dtype columns is None.
            ticker = Column(object)

            # Use integers for integer-valued categorical data like sector or
            # industry codes. Integer-dtype columns require an explicit missing
            # value.
            sector_code = Column(int, missing_value=-1)

            # Use bool for boolean-valued flags. Note that the default missing
            # value for bool-dtype columns is False.
            is_primary_share = Column(bool)

    Notes
    -----
    Because numpy has no native support for integers with missing values, users
    are strongly encouraged to use floats for any data that's semantically
    numeric. Doing so enables the use of `NaN` as a natural missing value,
    which has useful propagation semantics.
    """

    domain = GENERIC
    ndim = 2

    @classmethod
    def get_column(cls, name):
        """Look up a column by name.

        Parameters
        ----------
        name : str
            Name of the column to look up.

        Returns
        -------
        column : zipline.pipeline.data.BoundColumn
            Column with the given name.

        Raises
        ------
        AttributeError
            If no column with the given name exists.
        """
        clsdict = vars(cls)
        try:
            maybe_column = clsdict[name]
            if not isinstance(maybe_column, _BoundColumnDescr):
                raise KeyError(name)
        except KeyError as exc:
            raise AttributeError(
                "{dset} has no column {colname!r}:\n\n"
                "Possible choices are:\n"
                "{choices}".format(
                    dset=cls.qualname,
                    colname=name,
                    choices=bulleted_list(
                        sorted(cls._column_names),
                        max_count=10,
                    ),
                )
            ) from exc

        # Resolve column descriptor into a BoundColumn.
        return maybe_column.__get__(None, cls)


# This attribute is set by DataSetMeta to mark that a class is the root of a
# family of datasets with diffent domains. We don't want that behavior for the
# base DataSet class, and we also don't want to accidentally use a shared
# version of this attribute if we fail to set this in a subclass somewhere.
del DataSet._domain_specializations


class DataSetFamilyLookupError(AttributeError):
    """Exception thrown when a column is accessed on a DataSetFamily
    instead of on the result of a slice.

    Parameters
    ----------
    family_name : str
        The name of the DataSetFamily on which the access occurred.
    column_name : str
        The name of the column accessed.
    """

    def __init__(self, family_name, column_name):
        self.family_name = family_name
        self.column_name = column_name

    def __str__(self):
        # NOTE: when ``aggregate`` is added, remember to update this message
        return dedent(
            """\
            Attempted to access column {c} from DataSetFamily {d}:

            To work with dataset families, you must first select a
            slice using the ``slice`` method:

                {d}.slice(...).{c}
            """.format(
                c=self.column_name, d=self.family_name
            )
        )


class _DataSetFamilyColumn:
    """Descriptor used to raise a helpful error when a column is accessed on a
    DataSetFamily instead of on the result of a slice.

    Parameters
    ----------
    column_names : str
        The name of the column.
    """

    def __init__(self, column_name):
        self.column_name = column_name

    def __get__(self, instance, owner):
        raise DataSetFamilyLookupError(
            owner.__name__,
            self.column_name,
        )


class DataSetFamilyMeta(abc.ABCMeta):
    def __new__(cls, name, bases, dict_):
        columns = {}
        for k, v in dict_.items():
            if isinstance(v, Column):
                # capture all the columns off the DataSetFamily class
                # and replace them with a descriptor that will raise a helpful
                # error message. The columns will get added to the BaseSlice
                # for this type.
                columns[k] = v
                dict_[k] = _DataSetFamilyColumn(k)

        is_abstract = dict_.pop("_abstract", False)

        self = super(DataSetFamilyMeta, cls).__new__(
            cls,
            name,
            bases,
            dict_,
        )

        if not is_abstract:
            self.extra_dims = extra_dims = OrderedDict(
                [(k, frozenset(v)) for k, v in OrderedDict(self.extra_dims).items()]
            )
            if not extra_dims:
                raise ValueError(
                    "DataSetFamily must be defined with non-empty"
                    " extra_dims, or with `_abstract = True`",
                )

            class BaseSlice(self._SliceType):
                dataset_family = self

                ndim = self.slice_ndim
                domain = self.domain

                locals().update(columns)

            BaseSlice.__name__ = "%sBaseSlice" % self.__name__
            self._SliceType = BaseSlice

        # each type gets a unique cache
        self._slice_cache = {}
        return self

    def __repr__(self):
        return "<DataSetFamily: %r, extra_dims=%r>" % (
            self.__name__,
            list(self.extra_dims),
        )


class DataSetFamilySlice(DataSet):
    """Marker type for slices of a
    :class:`zipline.pipeline.data.dataset.DataSetFamily` objects
    """


# XXX: This docstring was mostly written when the abstraction here was
# "MultiDimensionalDataSet". It probably needs some rewriting.
class DataSetFamily(metaclass=DataSetFamilyMeta):
    """
    Base class for Pipeline dataset families.

    Dataset families are used to represent data where the unique identifier for
    a row requires more than just asset and date coordinates. A
    :class:`DataSetFamily` can also be thought of as a collection of
    :class:`~zipline.pipeline.data.DataSet` objects, each of which has the same
    columns, domain, and ndim.

    :class:`DataSetFamily` objects are defined with one or more
    :class:`~zipline.pipeline.data.Column` objects, plus one additional field:
    ``extra_dims``.

    The ``extra_dims`` field defines coordinates other than asset and date that
    must be fixed to produce a logical timeseries. The column objects determine
    columns that will be shared by slices of the family.

    ``extra_dims`` are represented as an ordered dictionary where the keys are
    the dimension name, and the values are a set of unique values along that
    dimension.

    To work with a :class:`DataSetFamily` in a pipeline expression, one must
    choose a specific value for each of the extra dimensions using the
    :meth:`~zipline.pipeline.data.DataSetFamily.slice` method.
    For example, given a :class:`DataSetFamily`:

    .. code-block:: python

       class SomeDataSet(DataSetFamily):
           extra_dims = [
               ('dimension_0', {'a', 'b', 'c'}),
               ('dimension_1', {'d', 'e', 'f'}),
           ]

           column_0 = Column(float)
           column_1 = Column(bool)

    This dataset might represent a table with the following columns:

    ::

      sid :: int64
      asof_date :: datetime64[ns]
      timestamp :: datetime64[ns]
      dimension_0 :: str
      dimension_1 :: str
      column_0 :: float64
      column_1 :: bool

    Here we see the implicit ``sid``, ``asof_date`` and ``timestamp`` columns
    as well as the extra dimensions columns.

    This :class:`DataSetFamily` can be converted to a regular :class:`DataSet`
    with:

    .. code-block:: python

       DataSetSlice = SomeDataSet.slice(dimension_0='a', dimension_1='e')

    This sliced dataset represents the rows from the higher dimensional dataset
    where ``(dimension_0 == 'a') & (dimension_1 == 'e')``.
    """

    _abstract = True  # Removed by metaclass

    domain = GENERIC
    slice_ndim = 2

    _SliceType = DataSetFamilySlice

    @type.__call__
    class extra_dims:
        """OrderedDict[str, frozenset] of dimension name -> unique values

        May be defined on subclasses as an iterable of pairs: the
        metaclass converts this attribute to an OrderedDict.
        """

        __isabstractmethod__ = True

        def __get__(self, instance, owner):
            return []

    @classmethod
    def _canonical_key(cls, args, kwargs):
        extra_dims = cls.extra_dims
        dimensions_set = set(extra_dims)
        if not set(kwargs) <= dimensions_set:
            extra = sorted(set(kwargs) - dimensions_set)
            raise TypeError(
                "%s does not have the following %s: %s\n"
                "Valid dimensions are: %s"
                % (
                    cls.__name__,
                    s("dimension", extra),
                    ", ".join(extra),
                    ", ".join(extra_dims),
                ),
            )

        if len(args) > len(extra_dims):
            raise TypeError(
                "%s has %d extra %s but %d %s given"
                % (
                    cls.__name__,
                    len(extra_dims),
                    s("dimension", extra_dims),
                    len(args),
                    plural("was", "were", args),
                ),
            )

        missing = object()
        coords = OrderedDict(zip(extra_dims, repeat(missing)))
        to_add = dict(zip(extra_dims, args))
        coords.update(to_add)
        added = set(to_add)

        for key, value in kwargs.items():
            if key in added:
                raise TypeError(
                    "%s got multiple values for dimension %r"
                    % (
                        cls.__name__,
                        coords,
                    ),
                )
            coords[key] = value
            added.add(key)

        missing = {k for k, v in coords.items() if v is missing}
        if missing:
            missing = sorted(missing)
            raise TypeError(
                "no coordinate provided to %s for the following %s: %s"
                % (
                    cls.__name__,
                    s("dimension", missing),
                    ", ".join(missing),
                ),
            )

        # validate that all of the provided values exist along their given
        # dimensions
        for key, value in coords.items():
            if value not in cls.extra_dims[key]:
                raise ValueError(
                    "%r is not a value along the %s dimension of %s"
                    % (
                        value,
                        key,
                        cls.__name__,
                    ),
                )

        return coords, tuple(coords.items())

    @classmethod
    def _make_dataset(cls, coords):
        """Construct a new dataset given the coordinates."""

        class Slice(cls._SliceType):
            extra_coords = coords

        Slice.__name__ = "%s.slice(%s)" % (
            cls.__name__,
            ", ".join("%s=%r" % item for item in coords.items()),
        )
        return Slice

    @classmethod
    def slice(cls, *args, **kwargs):
        """Take a slice of a DataSetFamily to produce a dataset
        indexed by asset and date.

        Parameters
        ----------
        *args
        **kwargs
            The coordinates to fix along each extra dimension.

        Returns
        -------
        dataset : DataSet
            A regular pipeline dataset indexed by asset and date.

        Notes
        -----
        The extra dimensions coords used to produce the result are available
        under the ``extra_coords`` attribute.
        """
        coords, hash_key = cls._canonical_key(args, kwargs)
        try:
            return cls._slice_cache[hash_key]
        except KeyError:
            pass

        Slice = cls._make_dataset(coords)
        cls._slice_cache[hash_key] = Slice
        return Slice


CurrencyConversion = namedtuple(
    "CurrencyConversion",
    ["currency", "field"],
)
