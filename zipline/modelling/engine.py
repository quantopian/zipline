"""
Compute Engine for FFC API
"""
from functools import wraps
from itertools import chain

from networkx import (
    DiGraph,
    get_node_attributes,
    topological_sort,
)


def build_dependency_graph(filters, classifiers, factors):
    dependencies = DiGraph()
    parents = set()
    for term in chain(filters, classifiers, factors):
        term._update_dependency_graph(
            dependencies,
            parents,
            parent_lookback=0,
        )

    # No parents should be left at the end of the run.
    assert not parents
    return dependencies


def require_thawed(method):
    """
    Decorator for FFCEngine methods that should only be called when the engine
    is not yet frozen.

    Usage:

    @require_thawed
    def some_method(self, arg):
        # Do stuff that should only be allowed if we're not yet frozen.
    """
    @wraps(method)
    def assert_not_frozen_then_call_method(self, *args, **kwargs):
        if self._frozen:
            raise AssertionError(
                "Can't call {cls}.{method_name} once frozen!".format(
                    cls=type(self).__name__, method_name=method.__name__,
                )
            )
        return method(self, *args, **kwargs)
    return assert_not_frozen_then_call_method


def require_frozen(method):
    """
    Decorator for FFCEngine methods that should only be called once the engine
    has been frozen.

    Usage:

    @require_frozen
    def some_method(self, arg):
        # Do stuff that should only be allowed once we're frozen
    """
    @wraps(method)
    def assert_frozen_then_call_method(self, *args, **kwargs):
        if not self._frozen:
            raise AssertionError(
                "Can't call {cls}.{method_name} until frozen!".format(
                    cls=type(self).__name__, method_name=method.__name__,
                )
            )
        return method(self, *args, **kwargs)
    return assert_frozen_then_call_method


class SimpleFFCEngine(object):
    """
    Class capable of computing terms of an FFC dependency graph.
    """
    __slots__ = [
        '_loader',
        '_trading_days',
        '_filters',
        '_classifiers',
        '_factors',
        '_frozen',
        '_graph',
        '_resolution_order',
        '_lookbacks',
    ]

    def __init__(self, loader, trading_days, asset_metadata):

        self._loader = loader
        self._trading_days = trading_days

        self._filters = []
        self._classifiers = []
        self._factors = []

        # These are set once all FFC terms have been added and self.freeze() is
        # called.
        self._frozen = False
        self._graph = None
        self._resolution_order = None
        self._lookbacks = None

    @require_thawed
    def add_filter(self, filter):
        self._filters.append(filter)

    @require_thawed
    def add_factor(self, factor):
        self._factors.append(factor)

    @require_thawed
    def add_classifier(self, classifier):
        self._classifiers.append(classifier)

    @require_thawed
    def freeze(self):
        """
        Called by TradingAlgorithm to signify that no more FFC Terms will be
        added to the graph.
        """
        self._filters = frozenset(self._filters)
        self._classifiers = frozenset(self._classifiers)
        self._factors = frozenset(self._factors)

        self._graph = build_dependency_graph(
            self._filters,
            self._classifiers,
            self._factors,
        )
        self._resolution_order = topological_sort(self._graph)
        self._lookbacks = get_node_attributes(self._graph, 'lookback')
        self._frozen = True

    @require_frozen
    def lookback(self, term):
        """
        Get the amount of lookback needed by parents of this term.
        """
        return self._lookbacks[term]

    @require_frozen
    def compute_chunk(self, dates, assets):
        """
        Compute our factors on a chunk of assets and dates.
        """
        loader = self._loader
        workspace = {term: None for term in self._resolution_order}

        for term in self._resolution_order:

            # TODO: Extend dates backward based on lookback.
            lookback = 5

            # Potential Optimization: Scan the resolution order for terms in
            # the same dataset and load them here as well.
            if term.atomic:
                workspace[term] = loader.load_adjusted_array(
                    [term],
                    dates,
                    assets,
                    lookback,
                )[0]
            else:
                workspace[term] = term.compute_chunk(
                    assets,
                    dates,
                    [workspace[input_] for input_ in term.inputs],
                )

        return workspace
