"""
Base class for Filters, Factors and Classifiers
"""
from abc import ABCMeta, abstractproperty
from weakref import WeakValueDictionary

from numpy import dtype as dtype_class
from six import with_metaclass

from zipline.errors import (
    DTypeNotSpecified,
    InputTermNotAtomic,
    InvalidDType,
    TermInputsNotSpecified,
    WindowLengthNotSpecified,
)
from zipline.utils.memoize import lazyval
from zipline.utils.numpy_utils import bool_dtype, default_fillvalue_for_dtype
from zipline.utils.sentinel import sentinel


NotSpecified = sentinel(
    'NotSpecified',
    'Singleton sentinel value used for Term defaults.',
)


class Term(with_metaclass(ABCMeta, object)):
    """
    Base class for terms in a Pipeline API compute graph.
    """
    # These are NotSpecified because a subclass is required to provide them.
    dtype = NotSpecified
    domain = NotSpecified

    # Subclasses aren't required to provide `params`.  The default behavior is
    # no params.
    params = ()

    _term_cache = WeakValueDictionary()

    def __new__(cls,
                domain=domain,
                dtype=dtype,
                # params is explicitly not allowed to be passed to an instance.
                *args,
                **kwargs):
        """
        Memoized constructor for Terms.

        Caching previously-constructed Terms is useful because it allows us to
        only compute equivalent sub-expressions once when traversing a Pipeline
        dependency graph.

        Caching previously-constructed Terms is **sane** because terms and
        their inputs are both conceptually immutable.
        """
        # Class-level attributes can be used to provide defaults for Term
        # subclasses.

        if domain is NotSpecified:
            domain = cls.domain

        dtype = cls._validate_dtype(dtype)
        params = cls._pop_params(kwargs)

        identity = cls.static_identity(
            domain=domain,
            dtype=dtype,
            params=params,
            *args, **kwargs
        )

        try:
            return cls._term_cache[identity]
        except KeyError:
            new_instance = cls._term_cache[identity] = \
                super(Term, cls).__new__(cls)._init(
                    domain=domain,
                    dtype=dtype,
                    params=params,
                    *args, **kwargs
                )
            return new_instance

    @classmethod
    def _pop_params(cls, kwargs):
        """
        Pop entries from the `kwargs` passed to cls.__new__ based on the values
        in `cls.params`.

        Parameters
        ----------
        kwargs : dict
            The kwargs passed to cls.__new__.

        Returns
        -------
        params : list[(str, object)]
            A list of string, value pairs containing the entries in cls.params.

        Raises
        ------
        TypeError
            Raised if any parameter values are not passed or not hashable.
        """
        param_values = []
        for key in cls.params:
            try:
                value = kwargs.pop(key)
                # Check here that the value is hashable so that we fail here
                # instead of trying to hash the param values tuple later.
                hash(key)
                param_values.append(value)
            except KeyError:
                raise TypeError(
                    "{typename} expected a keyword parameter {name!r}.".format(
                        typename=cls.__name__,
                        name=key
                    )
                )
            except TypeError:
                # Value wasn't hashable.
                raise TypeError(
                    "{typename} expected a hashable value for parameter "
                    "{name!r}, but got {value!r} instead.".format(
                        typename=cls.__name__,
                        name=key,
                        value=value,
                    )
                )
        return tuple(zip(cls.params, param_values))

    @classmethod
    def _validate_dtype(cls, passed_dtype):
        """
        Validate a `dtype` passed to Term.__new__.

        If passed_dtype is NotSpecified, then we try to fall back to a
        class-level attribute.  If a value is found at that point, we pass it
        to np.dtype so that users can pass `float` or `bool` and have them
        coerce to the appropriate numpy types.

        Returns
        -------
        validated : np.dtype
            The dtype to use for the new term.

        Raises
        ------
        DTypeNotSpecified
            When no dtype was passed to the instance, and the class doesn't
            provide a default.
        InvalidDType
            When either the class or the instance provides a value not
            coercible to a numpy dtype.
        """
        dtype = passed_dtype
        if dtype is NotSpecified:
            dtype = cls.dtype
        if dtype is NotSpecified:
            raise DTypeNotSpecified(termname=cls.__name__)
        try:
            dtype = dtype_class(dtype)
        except TypeError:
            raise InvalidDType(dtype=dtype, termname=cls.__name__)
        return dtype

    def __init__(self, *args, **kwargs):
        """
        Noop constructor to play nicely with our caching __new__.  Subclasses
        should implement _init instead of this method.

        When a class' __new__ returns an instance of that class, Python will
        automatically call __init__ on the object, even if a new object wasn't
        actually constructed.  Because we memoize instances, we often return an
        object that was already initialized from __new__, in which case we
        don't want to call __init__ again.

        Subclasses that need to initialize new instances should override _init,
        which is guaranteed to be called only once.
        """
        pass

    @classmethod
    def static_identity(cls, domain, dtype, params):
        """
        Return the identity of the Term that would be constructed from the
        given arguments.

        Identities that compare equal will cause us to return a cached instance
        rather than constructing a new one.  We do this primarily because it
        makes dependency resolution easier.

        This is a classmethod so that it can be called from Term.__new__ to
        determine whether to produce a new instance.
        """
        return (cls, domain, dtype, params)

    def _init(self, domain, dtype, params):
        """
        Parameters
        ----------
        domain : object
            Unused placeholder.
        dtype : np.dtype
            Dtype of this term's output.
        params : tuple[(str, hashable)]
            Tuple of key/value pairs of additional parameters.
        """
        self.domain = domain
        self.dtype = dtype

        for name, value in params:
            if hasattr(self, name):
                raise TypeError(
                    "Parameter {name!r} conflicts with already-present"
                    "attribute with value {value!r}.".format(
                        name=name,
                        value=getattr(self, name),
                    )
                )
            # TODO: Consider setting these values as attributes and replacing
            # the boilerplate in NumericalExpression, Rank, and
            # PercentileFilter.

        self.params = dict(params)

        # Make sure that subclasses call super() in their _validate() methods
        # by setting this flag.  The base class implementation of _validate
        # should set this flag to True.
        self._subclass_called_super_validate = False
        self._validate()
        del self._subclass_called_super_validate

        return self

    def _validate(self):
        """
        Assert that this term is well-formed.  This should be called exactly
        once, at the end of Term._init().
        """
        # mark that we got here to enforce that subclasses overriding _validate
        # call super().
        self._subclass_called_super_validate = True

    @abstractproperty
    def inputs(self):
        """
        A tuple of other Terms that this Term requires for computation.
        """
        raise NotImplementedError()

    @abstractproperty
    def mask(self):
        """
        A 2D Filter representing asset/date pairs to include while
        computing this Term. (True means include; False means exclude.)
        """
        raise NotImplementedError()

    @lazyval
    def dependencies(self):
        return self.inputs + (self.mask,)

    @lazyval
    def atomic(self):
        return not any(dep for dep in self.dependencies
                       if dep is not AssetExists())

    @lazyval
    def missing_value(self):
        return default_fillvalue_for_dtype(self.dtype)


class AssetExists(Term):
    """
    Pseudo-filter describing whether or not an asset existed on a given day.
    This is the default mask for all terms that haven't been passed a mask
    explicitly.

    This is morally a Filter, in the sense that it produces a boolean value for
    every asset on every date.  We don't subclass Filter, however, because
    `AssetExists` is computed directly by the PipelineEngine.

    See Also
    --------
    zipline.assets.AssetFinder.lifetimes
    """
    dtype = bool_dtype
    dataset = None
    extra_input_rows = 0
    inputs = ()
    dependencies = ()
    mask = None

    def __repr__(self):
        return "AssetExists()"


class CompositeTerm(Term):
    inputs = NotSpecified
    window_length = NotSpecified
    mask = NotSpecified

    def __new__(cls,
                inputs=inputs,
                window_length=window_length,
                mask=mask,
                *args, **kwargs):

        if inputs is NotSpecified:
            inputs = cls.inputs

        # Having inputs = NotSpecified is an error, but we handle it later
        # in self._validate rather than here.
        if inputs is not NotSpecified:
            # Allow users to specify lists as class-level defaults, but
            # normalize to a tuple so that inputs is hashable.
            inputs = tuple(inputs)

        if mask is NotSpecified:
            mask = cls.mask
        if mask is NotSpecified:
            mask = AssetExists()

        if window_length is NotSpecified:
            window_length = cls.window_length

        return super(CompositeTerm, cls).__new__(cls, inputs=inputs, mask=mask,
                                                 window_length=window_length,
                                                 *args, **kwargs)

    def _init(self, inputs, window_length, mask, *args, **kwargs):
        self.inputs = inputs
        self.window_length = window_length
        self.mask = mask
        return super(CompositeTerm, self)._init(*args, **kwargs)

    @classmethod
    def static_identity(cls, inputs, window_length, mask, *args, **kwargs):
        return (
            super(CompositeTerm, cls).static_identity(*args, **kwargs),
            inputs,
            window_length,
            mask,
        )

    def _validate(self):
        """
        Assert that this term is well-formed.  This should be called exactly
        once, at the end of Term._init().
        """
        if self.inputs is NotSpecified:
            raise TermInputsNotSpecified(termname=type(self).__name__)
        if self.window_length is NotSpecified:
            raise WindowLengthNotSpecified(termname=type(self).__name__)
        if self.mask is NotSpecified:
            # This isn't user error, this is a bug in our code.
            raise AssertionError("{term} has no mask".format(term=self))

        if self.window_length:
            for child in self.inputs:
                if not child.atomic:
                    raise InputTermNotAtomic(parent=self, child=child)

        return super(CompositeTerm, self)._validate()

    def _compute(self, inputs, dates, assets, mask):
        """
        Subclasses should implement this to perform actual computation.
        This is `_compute` rather than just `compute` because `compute` is
        reserved for user-supplied functions in CustomFactor.
        """
        raise NotImplementedError()

    @lazyval
    def windowed(self):
        """
        Whether or not this term represents a trailing window computation.

        If term.windowed is truthy, its compute_from_windows method will be
        called with instances of AdjustedArray as inputs.

        If term.windowed is falsey, its compute_from_baseline will be called
        with instances of np.ndarray as inputs.
        """
        return (
            self.window_length is not NotSpecified
            and self.window_length > 0
        )

    @lazyval
    def extra_input_rows(self):
        """
        The number of extra rows needed for each of our inputs to compute this
        term.
        """
        return max(0, self.window_length - 1)

    def __repr__(self):
        return (
            "{type}({inputs}, window_length={window_length})"
        ).format(
            type=type(self).__name__,
            inputs=self.inputs,
            window_length=self.window_length,
        )
