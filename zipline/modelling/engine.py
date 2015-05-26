"""
Compute Engine for FFC API
"""
from functools import wraps
from itertools import (
    chain,
    izip_longest,
)

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
            extra_rows=0,
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
    FFC Engine class that computes each term independently.
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
        '_extra_row_counts',
        '_max_extra_row_count',
    ]

    def __init__(self, loader, trading_days):

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
        self._extra_row_counts = None

    # Thawed Methods
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

        This is where we determine in what order and with what lookbacks we
        will compute our dependencies.
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

        self._extra_row_counts = get_node_attributes(self._graph, 'extra_rows')
        self._max_extra_row_count = max(self._extra_row_counts.values())

        self._frozen = True

    # Frozen Methods
    @require_frozen
    def extra_row_count(self, term):
        """
        Get the number extra rows to compute for the given term.
        """
        return self._extra_row_counts[term]

    @require_frozen
    def compute_chunk(self, start_date, end_date, assets):
        """
        Compute our factors on a chunk of assets and dates.
        """
        loader = self._loader
        trading_days = self._trading_days
        workspace = {term: None for term in self._resolution_order}

        start_idx = trading_days.get_loc(start_date)
        # +1 because we use this as the upper bound of a slice.
        end_idx = trading_days.get_loc(end_date) + 1
        if start_idx < self._max_extra_row_count:
            # TODO: Use NoFurtherDataError from trading.py
            raise ValueError(
                "Insufficient data to compute FFC Matrix: "
                "start date was %s, "
                "earliest known date was %s, "
                "Required extra rows was %d" % (
                    start_date, trading_days[0], self._max_extra_row_count,
                ),
            )

        all_dates = trading_days[start_idx - self._max_extra_row_count:end_idx]

        for term in self._resolution_order:
            # len(term_dates) == (end_idx - start_idx) + extra_row_count(term)
            term_start = self._max_extra_row_count - self.extra_row_count(term)
            term_dates = all_dates[term_start:]

            if term.atomic:
                # FUTURE OPTIMIZATION: Scan the resolution order for terms in
                # the same dataset and load them here as well.
                to_load = [term]
                loaded = loader.load_adjusted_array(
                    to_load,
                    term_dates,
                    assets,
                )
                for loaded_term, adj_array in izip_longest(to_load, loaded):
                    workspace[loaded_term] = adj_array
            else:
                workspace[term] = term._compute_chunk(
                    term_dates,
                    assets,
                    [workspace[input_] for input_ in term.inputs],
                )

        return workspace
