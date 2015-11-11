"""
Base class for Filters, Factors and Classifiers
"""
from abc import ABCMeta, abstractproperty
from weakref import WeakValueDictionary

from numpy import bool_, full, nan
from six import with_metaclass

from zipline.errors import (
    DTypeNotSpecified,
    InputTermNotAtomic,
    TermInputsNotSpecified,
    WindowLengthNotPositive,
    WindowLengthNotSpecified,
)
from zipline.utils.memoize import lazyval
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

    _term_cache = WeakValueDictionary()

    def __new__(cls,
                domain=NotSpecified,
                dtype=NotSpecified,
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

        if dtype is NotSpecified:
            dtype = cls.dtype

        identity = cls.static_identity(
            domain=domain,
            dtype=dtype,
            *args, **kwargs
        )

        try:
            return cls._term_cache[identity]
        except KeyError:
            new_instance = cls._term_cache[identity] = \
                super(Term, cls).__new__(cls)._init(
                    domain=domain,
                    dtype=dtype,
                    *args, **kwargs
                )
            return new_instance

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

    def _init(self, domain, dtype):
        self.domain = domain
        self.dtype = dtype

        self._validate()
        return self

    @classmethod
    def static_identity(cls, domain, dtype):
        """
        Return the identity of the Term that would be constructed from the
        given arguments.

        Identities that compare equal will cause us to return a cached instance
        rather than constructing a new one.  We do this primarily because it
        makes dependency resolution easier.

        This is a classmethod so that it can be called from Term.__new__ to
        determine whether to produce a new instance.
        """
        return (cls, domain, dtype)

    def _validate(self):
        """
        Assert that this term is well-formed.  This should be called exactly
        once, at the end of Term._init().
        """
        if self.dtype is NotSpecified:
            raise DTypeNotSpecified(termname=type(self).__name__)

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
    dtype = bool_
    dataset = None
    extra_input_rows = 0
    inputs = ()
    dependencies = ()
    mask = None

    def __repr__(self):
        return "AssetExists()"


# TODO: Move mixins to a separate file?
class SingleInputMixin(object):

    def _validate(self):
        num_inputs = len(self.inputs)
        if num_inputs != 1:
            raise ValueError(
                "{typename} expects only one input, "
                "but received {num_inputs} instead.".format(
                    typename=type(self).__name__,
                    num_inputs=num_inputs
                )
            )
        return super(SingleInputMixin, self)._validate()


class RequiredWindowLengthMixin(object):
    def _validate(self):
        if not self.windowed:
            raise WindowLengthNotPositive(window_length=self.window_length)
        return super(RequiredWindowLengthMixin, self)._validate()


class CustomTermMixin(object):
    """
    Mixin for user-defined rolling-window Terms.

    Implements `_compute` in terms of a user-defined `compute` function, which
    is mapped over the input windows.

    Used by CustomFactor, CustomFilter, CustomClassifier, etc.
    """

    def __new__(cls, inputs=NotSpecified, window_length=NotSpecified):

        return super(CustomTermMixin, cls).__new__(
            cls,
            inputs=inputs,
            window_length=window_length,
        )

    def __init__(self, inputs=NotSpecified, window_length=NotSpecified):
        return super(CustomTermMixin, self).__init__(
            inputs=inputs,
            window_length=window_length,
        )

    def compute(self, today, assets, out, *arrays):
        """
        Override this method with a function that writes a value into `out`.
        """
        raise NotImplementedError()

    def _compute(self, windows, dates, assets, mask):
        """
        Call the user's `compute` function on each window with a pre-built
        output array.
        """
        # TODO: Make mask available to user's `compute`.
        compute = self.compute
        out = full(mask.shape, nan, dtype=self.dtype)
        with self.ctx:
            # TODO: Consider pre-filtering columns that are all-nan at each
            # time-step?
            for idx, date in enumerate(dates):
                compute(
                    date,
                    assets,
                    out[idx],
                    *(next(w) for w in windows)
                )
        out[~mask] = nan
        return out

    def short_repr(self):
        return type(self).__name__ + '(%d)' % self.window_length


class CompositeTerm(Term):
    inputs = NotSpecified
    window_length = NotSpecified
    mask = NotSpecified

    def __new__(cls, inputs=NotSpecified, window_length=NotSpecified,
                mask=NotSpecified, *args, **kwargs):

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
