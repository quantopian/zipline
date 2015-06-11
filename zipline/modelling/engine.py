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
from numpy import (
    empty,
)

from zipline.data.adjusted_array import ensure_ndarray


def build_dependency_graph(filters, classifiers, factors):
    """
    Build a dependency graph containing the given filters, classifiers,
    factors, and all their dependencies.

    Parameters
    ----------
    filters: list
        A list of Filter objects.
    classifiers: list
        A list of Classifier objects.
    factors: list
        A list of Factor objects.

    Returns
    -------
    dependencies : networkx.DiGraph
        A directed graph representing the dependencies of the desired inputs.
    """
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

    Usage
    -----

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

    Usage
    -----

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
        '_calendar',
        '_filters',
        '_classifiers',
        '_factors',
        '_frozen',
        '_graph',
        '_resolution_order',
        '_extra_row_counts',
    ]

    def __init__(self, loader, calendar):

        self._loader = loader
        self._calendar = calendar

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

        This is where we determine in what order and with what window lengths
        we will compute our dependencies.
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
        self._frozen = True

    # Frozen Methods
    @require_frozen
    def extra_row_count(self, term):
        """
        Get the number extra rows to compute for the given term.
        """
        return self._extra_row_counts[term]

    def _date_slice_bounds(self, start_date, end_date):
        """
        Helper for compute_chunk.

        Get indices (start_idx, end_idx) from our calendar such that:
        self._calendar[start_idx:end_idx] returns dates between start_date and
        end_date, inclusive.
        """
        calendar = self._calendar
        start_date_idx = calendar.get_loc(start_date)
        end_date_idx = calendar.get_loc(end_date)
        max_extra_rows = max(self._extra_row_counts.values())
        if start_date_idx < max_extra_rows:
            # TODO: Use NoFurtherDataError from trading.py
            raise ValueError(
                "Insufficient data to compute FFC Matrix: "
                "start date was %s, "
                "earliest known date was %s, "
                "Required extra rows was %d" % (
                    start_date, calendar[0], max_extra_rows,
                ),
            )
        # Increment end_date_idx by 1 so that slicing with [start:end] includes
        # end_date.
        return start_date_idx, end_date_idx + 1

    @require_frozen
    def _inputs_for_term(self, term, workspace, windowed):
        """
        Compute inputs for the given term.

        This is mostly complicated by the fact that for each input we store
        as many rows as will be necessary to serve any term requiring that
        input.  Thus if Factor A needs 5 extra rows of price, and Factor B
        needs 3 extra rows of price, we need to remove 2 leading rows from our
        stored prices before passing them to Factor B.
        """
        term_extra_rows = term.extra_input_rows
        if windowed:
            return [
                workspace[input_].traverse(
                    term.window_length,
                    offset=self.extra_row_count(input_) - term_extra_rows
                )
                for input_ in term.inputs
            ]
        else:
            return [
                ensure_ndarray(
                    workspace[input_][
                        self.extra_row_count(input_) - term_extra_rows:
                    ],
                )
                for input_ in term.inputs
            ]

    @require_frozen
    def compute_chunk(self, start_date, end_date, assets):
        """
        Compute our factors on a chunk of assets and dates.
        """
        loader = self._loader
        calendar = self._calendar
        workspace = {term: None for term in self._resolution_order}
        start_idx, end_idx = self._date_slice_bounds(start_date, end_date)

        for term in self._resolution_order:
            dates = calendar[start_idx - self.extra_row_count(term):end_idx]
            if term.atomic:
                # FUTURE OPTIMIZATION: Scan the resolution order for terms in
                # the same dataset and load them here as well.
                to_load = [term]
                loaded = loader.load_adjusted_array(
                    to_load,
                    dates,
                    assets,
                )
                for loaded_term, adj_array in izip_longest(to_load, loaded):
                    workspace[loaded_term] = adj_array
            elif term.windowed:
                workspace[term] = empty(
                    shape=(len(dates), len(assets)),
                    dtype=term.dtype
                )
                term.compute_from_windows(
                    self._inputs_for_term(term, workspace, windowed=True),
                    workspace[term],
                    dates,
                    assets,
                )
            else:
                workspace[term] = empty(
                    shape=(len(dates), len(assets)),
                    dtype=term.dtype
                )
                term.compute_from_arrays(
                    self._inputs_for_term(term, workspace, windowed=False),
                    workspace[term],
                    dates,
                    assets,
                )

        return workspace
