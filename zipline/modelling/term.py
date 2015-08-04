"""
Base class for Filters, Factors and Classifiers
"""
from numpy import (
    empty,
    float64,
    full,
    nan,
)
from weakref import WeakValueDictionary

from zipline.errors import (
    InputTermNotAtomic,
    TermInputsNotSpecified,
    WindowLengthNotPositive,
    WindowLengthNotSpecified,
)
from zipline.utils.lazyval import lazyval


NotSpecified = (object(),)


class Term(object):
    """
    Base class for terms in an FFC API compute graph.
    """
    inputs = NotSpecified
    window_length = NotSpecified
    domain = None
    dtype = float64

    _term_cache = WeakValueDictionary()

    def __new__(cls,
                inputs=None,
                window_length=None,
                domain=None,
                dtype=None,
                *args,
                **kwargs):
        """
        Memoized constructor for Terms.

        Caching previously-constructed Terms is useful because it allows us to
        only compute equivalent sub-expressions once when traversing an FFC
        dependency graph.

        Caching previously-constructed Terms is **sane** because terms and
        their inputs are both conceptually immutable.
        """
        if inputs is None:
            inputs = tuple(cls.inputs)
        else:
            inputs = tuple(inputs)

        if window_length is None:
            window_length = cls.window_length

        if domain is None:
            domain = cls.domain

        if dtype is None:
            dtype = cls.dtype

        identity = cls.static_identity(
            inputs=inputs,
            window_length=window_length,
            domain=domain,
            dtype=dtype,
            *args, **kwargs
        )

        try:
            return cls._term_cache[identity]
        except KeyError:
            new_instance = cls._term_cache[identity] = \
                super(Term, cls).__new__(cls)._init(
                    inputs=inputs,
                    window_length=window_length,
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

    def _init(self, inputs, window_length, domain, dtype):
        self.inputs = inputs
        self.window_length = window_length
        self.domain = domain
        self.dtype = dtype

        self._validate()
        return self

    @classmethod
    def static_identity(cls, inputs, window_length, domain, dtype):
        """
        Return the identity of the Term that would be constructed from the
        given arguments.

        Identities that compare equal will cause us to return a cached instance
        rather than constructing a new one.  We do this primarily because it
        makes dependency resolution easier.

        This is a classmethod so that it can be called from Term.__new__ to
        determine whether to produce a new instance.
        """
        return (cls, inputs, window_length, domain, dtype)

    def _validate(self):
        """
        Assert that this term is well-formed.  This should be called exactly
        once, at the end of Term._init().
        """
        if self.inputs is NotSpecified:
            raise TermInputsNotSpecified(termname=type(self).__name__)
        if self.window_length is NotSpecified:
            raise WindowLengthNotSpecified(termname=type(self).__name__)

        if self.window_length:
            for child in self.inputs:
                if not child.atomic:
                    raise InputTermNotAtomic(parent=self, child=child)

    @lazyval
    def atomic(self):
        """
        Whether or not this term has dependencies.

        If term.atomic is truthy, it should have dataset and dtype attributes.
        """
        return len(self.inputs) == 0

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

    def compute_from_windows(self, windows, mask):
        """
        Subclasses should implement this for computations requiring moving
        windows of continually-adjusting data.
        """
        raise NotImplementedError()

    def compute_from_arrays(self, arrays, mask):
        """
        Subclasses should implement this for computations that can be expressed
        directly as array computations.
        """
        raise NotImplementedError()

    def __repr__(self):
        return (
            "{type}({inputs}, window_length={window_length})"
        ).format(
            type=type(self).__name__,
            inputs=self.inputs,
            window_length=self.window_length,
        )


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
        if self.windowed or self.window_length is NotSpecified:
            return super(RequiredWindowLengthMixin, self)._validate()
        raise WindowLengthNotPositive(window_length=self.window_length)


class CustomTermMixin(object):
    """
    Mixin for user-defined rolling-window Terms.

    Implements `compute_from_windows` in terms of a user-defined `compute`
    function, which is mapped over the input windows.

    Used by CustomFactor, CustomFilter, CustomClassifier, etc.
    """

    def compute(self, today, assets, out, *arrays):
        """
        Override this method with a function that writes a value into `out`.
        """
        raise NotImplementedError()

    def compute_from_windows(self, windows, mask):
        """
        Call the user's `compute` function on each window with a pre-built
        output array.
        """
        # TODO: Make mask available to user's `compute`.
        compute = self.compute
        dates, assets = mask.index, mask.columns
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
        out[~mask.values] = nan
        return out


class TestingTermMixin(object):
    """
    Mixin for Term subclasses testing engines that asserts all inputs are
    correctly shaped.

    Used by TestingTerm, TestingFilter, TestingClassifier, etc.
    """
    def compute_from_windows(self, windows, mask):
        assert self.window_length > 0
        dates, assets = mask.index, mask.columns
        outbuf = empty(mask.shape, dtype=self.dtype)
        for idx, _ in enumerate(dates):
            result = self.from_windows(*(next(w) for w in windows))
            assert result.shape == (len(assets),)
            outbuf[idx] = result

        for window in windows:
            try:
                next(window)
            except StopIteration:
                pass
            else:
                raise AssertionError("window %s was not exhausted" % window)
        return outbuf

    def compute_from_arrays(self, arrays, mask):
        assert self.window_length == 0
        outbuf = empty(mask.shape, dtype=self.dtype)
        for array in arrays:
            assert array.shape == outbuf.shape
        outbuf[:] = self.from_arrays(*arrays)
        return outbuf
