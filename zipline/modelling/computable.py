"""
Base class for Filters, Factors and Classifiers
"""
from numpy import float64


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
        return self

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
                                 parent_lookback):
        """
        Add this term and all its inputs to dependencies.
        """

        # If we've seen this node already as a parent of the current traversal,
        # it means we have an unsatisifiable dependency.
        if self in parents:
            raise CyclicDependency(self)
        parents.add(self)

        # If we're already in the graph because we're a dependency of a
        # processed node, ensure that we load enough lookback data for our
        # dependencies.
        try:
            existing = dependencies.node[self]
            existing['lookback'] = max(parent_lookback, existing['lookback'])
        except KeyError:
            dependencies.add_node(self, lookback=parent_lookback)

        for term in self.inputs:
            term._update_dependency_graph(
                dependencies,
                parents,
                parent_lookback=parent_lookback + self.lookback,
            )
            dependencies.add_edge(term, self)

        parents.remove(self)

    def compute_chunk(self, dates, assets, dependencies):
        """
        """
        lookback = self.lookback
        expected_shape = len(dates) + lookback, len(assets)
        for dep in dependencies:
            assert dep.shape == expected_shape

        for offset, date in enumerate(dates):
            self.compute_single_date(
                date,
                assets,
                *[dep[offset:lookback + offset, :] for dep in dependencies]
            )

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
