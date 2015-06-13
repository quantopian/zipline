"""
Base class for Filters, Factors and Classifiers
"""
from numpy import (
    float64,
)

from zipline.errors import (
    InputTermNotAtomic,
    TermInputsNotSpecified,
    WindowLengthNotSpecified,
)
from zipline.utils.lazyval import lazyval


class CyclicDependency(Exception):
    # TODO: Move this somewhere else.
    pass


NotSpecified = (object(),)


class Term(object):
    """
    Base class for terms in an FFC API compute graph.
    """
    inputs = NotSpecified
    window_length = NotSpecified
    domain = None
    dtype = float64

    _term_cache = {}

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
        # TODO: Default-construct instances when passed a class?
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
            new_instance = super(Term, cls).__new__(cls)._init(
                inputs=inputs,
                window_length=window_length,
                domain=domain,
                dtype=dtype,
                *args, **kwargs
            )
            cls._term_cache[identity] = new_instance
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
        return self.window_length > 0

    @lazyval
    def extra_input_rows(self):
        """
        The number of extra rows needed for each of our inputs to compute this
        term.
        """
        return max(0, self.window_length - 1)

    def _update_dependency_graph(self,
                                 dependencies,
                                 parents,
                                 extra_rows):
        """
        Add this term and all its inputs to dependencies.
        """
        # If we've seen this node already as a parent of the current traversal,
        # it means we have an unsatisifiable dependency.  This should only be
        # possible if the term's inputs are mutated after construction.
        if self in parents:
            raise CyclicDependency(self)
        parents.add(self)

        try:
            existing = dependencies.node[self]
        except KeyError:
            # We're not yet in the graph: add ourself with the specified number
            # of extra rows.
            dependencies.add_node(self, extra_rows=extra_rows)
        else:
            # We're already in the graph because we've been traversed by
            # another parent.  Ensure that we have enough extra rows to satisfy
            # all of our parents.
            existing['extra_rows'] = max(extra_rows, existing['extra_rows'])

        for term in self.inputs:
            term._update_dependency_graph(
                dependencies,
                parents,
                extra_rows=extra_rows + self.extra_input_rows,
            )
            dependencies.add_edge(term, self)

        parents.remove(self)

    def compute_from_windows(self, windows, outbuf, dates, assets):
        """
        Subclasses should implement this for computations requiring moving
        windows.
        """
        raise NotImplementedError()

    def compute_from_arrays(self, arrays, outbuf, dates, assets):
        """
        Subclasses should implement this for computations that can be expressed
        directly as array computations.
        """
        raise NotImplementedError()

    def __repr__(self):
        return (
            "{type}(inputs={inputs}, "
            "window_length={window_length})"
        ).format(
            type=type(self).__name__,
            inputs=self.inputs,
            window_length=self.window_length,
            domain=self.domain,
        )


class SingleInputMixin(object):

    def _validate(self):
        if len(self.inputs) != 1:
            raise ValueError("inputs must be of length 1")
        return super(SingleInputMixin, self)._validate()
