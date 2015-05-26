"""
Base class for Filters, Factors and Classifiers
"""
from numpy import (
    asarray,
    empty,
    float64,
)

from zipline.errors import InputTermNotAtomic


class CyclicDependency(Exception):
    # TODO: Move this somewhere else.
    pass


class Term(object):
    """
    Base class for terms in an FFC API compute graph.
    """
    inputs = ()
    lookback = 0
    domain = None
    dtype = float64

    _term_cache = {}

    def __new__(cls,
                inputs=None,
                lookback=None,
                domain=None,
                dtype=None,
                *args,
                **kwargs):
        """
        Memoized constructor for Terms.

        Caching previously-constructed Terms is useful because it allows us to
        only compute equivalent sub-expressions once when traversing an FFC
        dependency graph.

        Caching previously-constructed Terms is **sane** because they're
        conceptually immutable.
        """

        inputs = tuple(inputs or cls.inputs)
        lookback = lookback or cls.lookback
        domain = domain or cls.domain
        dtype = dtype or cls.dtype

        identity = cls.static_identity(
            inputs=inputs,
            lookback=lookback,
            domain=domain,
            dtype=dtype,
            *args, **kwargs
        )

        try:
            return cls._term_cache[identity]
        except KeyError:
            new_instance = super(Term, cls).__new__(cls)._init(
                inputs=inputs,
                lookback=lookback,
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

    def _init(self, inputs, lookback, domain, dtype):
        self.inputs = inputs
        self.lookback = lookback
        self.domain = domain
        self.dtype = dtype

        self._validate()
        return self

    def _validate(self):
        """
        Assert that this term is well-formed. This currently means the
        following:

        - If we have a lookback window, all of our inputs are atomic.
        """
        if self.lookback:
            for child in self.inputs:
                if not child.atomic:
                    raise InputTermNotAtomic(parent=self, child=child)

    @property
    def atomic(self):
        """
        Whether or not this term has dependencies.

        If term.atomic is truthy, it should have dataset and dtype attributes.
        """
        return len(self.inputs) == 0

    @classmethod
    def static_identity(cls, inputs, lookback, domain, dtype):
        """
        Return the identity of the Term that would be constructed from the
        given arguments.

        Identities that compare equal will cause us to return a cached instance
        rather than constructing a new one.  We do this primarily because it
        makes dependency resolution easier.

        This is a classmethod so that it can be called from Term.__new__ to
        determine whether to produce a new instance.
        """
        return (cls, inputs, lookback, domain, dtype)

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

        # Add ourself to the graph with the specified number of extra rows.  If
        # we're already in the graph because we have multiple parents, ensure
        # that we have enough extra rows to satisfy both of our parents.
        try:
            existing = dependencies.node[self]
        except KeyError:
            dependencies.add_node(self, extra_rows=extra_rows)
        else:
            existing['extra_rows'] = max(extra_rows, existing['extra_rows'])

        for term in self.inputs:
            term._update_dependency_graph(
                dependencies,
                parents,
                # Each lookback row after the first requires us to load/compute
                # an extra row from each of our dependencies.
                extra_rows=extra_rows + max(0, self.lookback - 1),
            )
            dependencies.add_edge(term, self)

        parents.remove(self)

    def _compute_chunk(self, dates, assets, dependencies):
        """
        Compute the given term for dates/assets.
        """
        lookback = self.lookback
        outbuf = empty(
            (len(dates), len(assets))
        )

        # Traverse trailing windows.
        if self.lookback:
            return self.compute_from_windows(
                outbuf,
                dates,
                assets,
                [dep.traverse(lookback) for dep in dependencies],
            )
        else:
            return self.compute_from_baselines(
                outbuf,
                dates,
                assets,
                [asarray(dep.data) for dep in dependencies],
            )

    def compute_from_windows(self, outbuf, dates, assets, windows):
        """
        Subclasses should implement this for computations requiring moving
        windows.
        """
        raise NotImplementedError()

    def compute_from_baselines(self, outbuf, dates, assets, arrays):
        """
        Subclasses should implement this for computations that can be expressed
        directly as array computations.
        """
        raise NotImplementedError()

    def __repr__(self):
        return (
            "{type}(inputs={inputs}, "
            "lookback={lookback}, domain={domain})"
        ).format(
            type=type(self).__name__,
            inputs=self.inputs,
            lookback=self.lookback,
            domain=self.domain,
        )
