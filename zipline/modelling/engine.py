"""
Compute Engine for FFC API
"""
from abc import (
    ABCMeta,
    abstractmethod,
)
from operator import and_
from six import (
    iteritems,
    with_metaclass,
)
from six.moves import (
    reduce,
    zip,
    zip_longest,
)

from networkx import (
    DiGraph,
    get_node_attributes,
    topological_sort,
)
from numpy import (
    add,
    empty_like,
)
from pandas import (
    DataFrame,
    date_range,
    MultiIndex,
)

from zipline.lib.adjusted_array import ensure_ndarray
from zipline.errors import NoFurtherDataError
from zipline.modelling.factor import Factor
from zipline.modelling.filter import Filter


# TODO: Move this somewhere else.
class CyclicDependency(Exception):
    pass


def build_dependency_graph(terms):
    """
    Build a dependency graph containing the given terms and their dependencies.

    Parameters
    ----------
    terms : iterable
        An iterable of zipline.modelling.term.Term.

    Returns
    -------
    dependencies : networkx.DiGraph
        A directed graph representing the dependencies of the desired inputs.

        Each node in the graph has an `extra_rows` attribute, indicating how
        many, if any, extra rows we should compute for the node.  Extra rows
        are most often needed when a term is an input to a rolling window
        computation.  For example, if we compute a 30 day moving average of
        price from day X to day Y, we need to load price data for the range
        from day (X - 29) to day Y.
    """
    dependencies = DiGraph()
    parents = set()
    for term in terms:
        _add_to_graph(
            term,
            dependencies,
            parents,
            extra_rows=0,
        )
        # No parents should be left between top-level terms.
        assert not parents
    return dependencies


def _add_to_graph(term,
                  dependencies,
                  parents,
                  extra_rows):
    """
    Add the term and all its inputs to dependencies.
    """
    # If we've seen this node already as a parent of the current traversal,
    # it means we have an unsatisifiable dependency.  This should only be
    # possible if the term's inputs are mutated after construction.
    if term in parents:
        raise CyclicDependency(term)
    parents.add(term)

    try:
        existing = dependencies.node[term]
    except KeyError:
        # We're not yet in the graph: add the term with the specified number of
        # extra rows.
        dependencies.add_node(term, extra_rows=extra_rows)
    else:
        # We're already in the graph because we've been traversed by
        # another parent.  Ensure that we have enough extra rows to satisfy
        # all of our parents.
        existing['extra_rows'] = max(extra_rows, existing['extra_rows'])

    for subterm in term.inputs:
        _add_to_graph(
            subterm,
            dependencies,
            parents,
            extra_rows=extra_rows + term.extra_input_rows,
        )
        dependencies.add_edge(subterm, term)

    parents.remove(term)


class FFCEngine(with_metaclass(ABCMeta)):

    @abstractmethod
    def factor_matrix(self, terms, start_date, end_date):
        """
        Compute values for `terms` between `start_date` and `end_date`.

        Returns a DataFrame with a MultiIndex of (date, asset) pairs on the
        index.  On each date, we return a row for each asset that passed all
        instances of `Filter` in `terms, and the columns of the returned frame
        will be the keys in `terms` whose values are instances of `Factor`.

        Parameters
        ----------
        terms : dict
            Map from str -> zipline.modelling.term.Term.
        start_date : datetime
            The first date of the matrix.
        end_date : datetime
            The last date of the matrix.

        Returns
        -------
        matrix : pd.DataFrame
            A matrix of factors
        """
        raise NotImplementedError("factor_matrix")


class NoOpFFCEngine(FFCEngine):
    """
    FFCEngine that doesn't do anything.
    """

    def factor_matrix(self, terms, start, end):
        return DataFrame(
            index=MultiIndex.from_product(
                [date_range(start=start, end=end, freq='D'), ()],
            ),
            columns=sorted(terms.keys())
        )


class SimpleFFCEngine(object):
    """
    FFC Engine class that computes each term independently.

    Parameters
    ----------
    loader : FFCLoader
        A loader to use to retrieve raw data for atomic terms.
    calendar : DatetimeIndex
        Array of dates to consider as trading days when computing a range
        between a fixed start and end.
    asset_finder : zipline.assets.AssetFinder
        An AssetFinder instance.  We depend on the AssetFinder to determine
        which assets are in the top-level universe at any point in time.
    """
    __slots__ = [
        '_loader',
        '_calendar',
        '_finder',
        '__weakref__',
    ]

    def __init__(self, loader, calendar, asset_finder):
        self._loader = loader
        self._calendar = calendar
        self._finder = asset_finder

    def factor_matrix(self, terms, start_date, end_date):
        """
        Compute a factor matrix.

        Parameters
        ----------
        terms : dict[str -> zipline.modelling.term.Term]
            Dict mapping term names to instances.  The supplied names are used
            as column names in our output frame.
        start_date : pd.Timestamp
            Start date of the computed matrix.
        end_date : pd.Timestamp
            End date of the computed matrix.

        The algorithm implemented here can be broken down into the following
        stages:

        0. Build a dependency graph of all terms in `terms`.  Topologically
        sort the graph to determine an order in which we can compute the terms.

        1. Ask our AssetFinder for a "lifetimes matrix", which should contain,
        for each date between start_date and end_date, a boolean value for each
        known asset indicating whether the asset existed on that date.

        2. Compute each term in the dependency order determined in (0), caching
        the results in a a dictionary to that they can be fed into future
        terms.

        3. For each date, determine the number of assets passing **all**
        filters. The sum, N, of all these values is the total number of rows in
        our output frame, so we pre-allocate an output array of length N for
        each factor in `terms`.

        4. Fill in the arrays allocated in (3) by copying computed values from
        our output cache into the corresponding rows.

        5. Stick the values computed in (4) into a DataFrame and return it.

        Step 0 is performed in `build_dependency_graph`.
        Step 1 is performed in `self.build_lifetimes_matrix`.
        Step 2 is performed in `self.compute_chunk`.
        Steps 3, 4, and 5 are performed in self._format_factor_matrix.

        See Also
        --------
        FFCEngine.factor_matrix
        """
        if end_date <= start_date:
            raise ValueError(
                "start_date must be before end_date \n"
                "start_date=%s, end_date=%s" % (start_date, end_date)
            )

        graph = build_dependency_graph(terms.values())
        ordered_terms = topological_sort(graph)
        extra_row_counts = get_node_attributes(graph, 'extra_rows')
        max_extra_rows = max(extra_row_counts.values())

        lifetimes = self.build_lifetimes_matrix(
            start_date,
            end_date,
            max_extra_rows,
        )
        lifetimes_between_dates = lifetimes[max_extra_rows:]

        dates = lifetimes_between_dates.index.values
        assets = lifetimes_between_dates.columns.values

        raw_outputs = self.compute_chunk(
            ordered_terms,
            extra_row_counts,
            lifetimes,
        )

        # We only need filters and factors to compute the final output matrix.
        raw_filters = [lifetimes_between_dates.values]
        raw_factors = []
        factor_names = []
        for name, term in iteritems(terms):
            extra = extra_row_counts[term]
            if isinstance(term, Factor):
                factor_names.append(name)
                raw_factors.append(raw_outputs[term][extra:])

            elif isinstance(term, Filter):
                raw_filters.append(raw_outputs[term][extra:])

        return self._format_factor_matrix(
            dates,
            assets,
            raw_filters,
            raw_factors,
            factor_names,
        )

    def build_lifetimes_matrix(self, start_date, end_date, extra_rows):
        """
        Compute a lifetimes matrix from our AssetFinder, then drop columns that
        didn't exist at all during the query dates.

        Parameters
        ----------
        start_date : pd.Timestamp
            Base start date for the matrix.
        end_date : pd.Timestamp
            End date for the matrix.
        extra_rows : int
            Number of rows prior to `start_date` to include.
            Extra rows are needed by terms like moving averages that require a
            trailing window of data to compute.

        Returns
        -------
        lifetimes : pd.DataFrame
            Frame of dtype `bool` containing dates from `extra_rows` days
            before `start_date`, continuing through to `end_date`.  The
            returned frame contains as columns all assets in our AssetFinder
            that existed for at least one day between `start_date` and
            `end_date`.
        """
        calendar = self._calendar
        finder = self._finder
        start_idx, end_idx = self._calendar.slice_locs(start_date, end_date)
        if start_idx < extra_rows:
            raise NoFurtherDataError(
                msg="Insufficient data to compute FFC Matrix: "
                "start date was %s, "
                "earliest known date was %s, "
                "and %d extra rows were requested." % (
                    start_date, calendar[0], extra_rows,
                ),
            )

        # Build lifetimes matrix reaching back as far start_date plus
        # max_extra_rows.
        lifetimes = finder.lifetimes(
            calendar[start_idx - extra_rows:end_idx]
        )
        assert lifetimes.index[extra_rows] == start_date
        assert lifetimes.index[-1] == end_date
        if not lifetimes.columns.unique:
            columns = lifetimes.columns
            duplicated = columns[columns.duplicated()].unique()
            raise AssertionError("Duplicated sids: %d" % duplicated)

        # Filter out columns that didn't exist between the requested start and
        # end dates.
        existed = lifetimes.iloc[extra_rows:].any()
        return lifetimes.loc[:, existed]

    def _inputs_for_term(self, term, workspace, extra_row_counts):
        """
        Compute inputs for the given term.

        This is mostly complicated by the fact that for each input we store
        as many rows as will be necessary to serve any term requiring that
        input.  Thus if Factor A needs 5 extra rows of price, and Factor B
        needs 3 extra rows of price, we need to remove 2 leading rows from our
        stored prices before passing them to Factor B.
        """
        term_extra_rows = term.extra_input_rows
        if term.windowed:
            return [
                workspace[input_].traverse(
                    term.window_length,
                    offset=extra_row_counts[input_] - term_extra_rows
                )
                for input_ in term.inputs
            ]
        else:
            return [
                ensure_ndarray(
                    workspace[input_][
                        extra_row_counts[input_] - term_extra_rows:
                    ],
                )
                for input_ in term.inputs
            ]

    def compute_chunk(self, ordered_terms, extra_row_counts, base_mask):
        """
        Compute the FFC terms in the graph based on the assets and dates
        defined by base_mask.

        Returns a dictionary mapping terms to computed arrays.
        """
        loader = self._loader
        max_extra_rows = max(extra_row_counts.values())
        workspace = {term: None for term in ordered_terms}

        for term in ordered_terms:
            base_mask_for_term = base_mask.iloc[
                max_extra_rows - extra_row_counts[term]:
            ]
            if term.atomic:
                # FUTURE OPTIMIZATION: Scan the resolution order for terms in
                # the same dataset and load them here as well.
                to_load = [term]
                loaded = loader.load_adjusted_array(
                    to_load,
                    base_mask_for_term,
                )
                for loaded_term, adj_array in zip_longest(to_load, loaded):
                    workspace[loaded_term] = adj_array
            else:
                if term.windowed:
                    compute = term.compute_from_windows
                else:
                    compute = term.compute_from_arrays
                workspace[term] = compute(
                    self._inputs_for_term(term, workspace, extra_row_counts),
                    base_mask_for_term,
                )
        return workspace

    def _format_factor_matrix(self,
                              dates,
                              assets,
                              filter_data,
                              factor_data,
                              factor_names):
        """
        Convert raw computed filters/factors into a DataFrame for public APIs.

        Parameters
        ----------
        dates : np.array[datetime64]
            Index for raw data in filter_data/factor_data.
        assets : np.array[int64]
            Column labels for raw data in filter_data/factor_data.
        filter_data : list[ndarray[bool]]
            Raw filters data.
        factor_data : list[ndarray]
            Raw factor data.
        factor_names : list[str]
            Names of factors to use as keys.

        Returns
        -------
        factor_matrix : pd.DataFrame
            A DataFrame with the following indices:

            index : two-tiered MultiIndex of (date, asset).  For each date, we
                return a row for each asset that passed all filters on that
                date.
            columns : keys from `factor_data`

        Each date/asset/factor triple contains the computed value of the given
        factor on the given date for the given asset.
        """
        # FUTURE OPTIMIZATION: Cythonize all of this.

        # Boolean mask of values that passed all filters.
        unioned = reduce(and_, filter_data)

        # Parallel arrays of (x,y) coords for all date/asset pairs that passed
        # all filters.  Each entry here will correspond to a row in our output
        # frame.
        nonzero_xs, nonzero_ys = unioned.nonzero()

        raw_dates_index = empty_like(nonzero_xs, dtype='datetime64[ns]')
        raw_assets_index = empty_like(nonzero_xs, dtype=int)
        factor_outputs = [
            empty_like(nonzero_xs, dtype=factor.dtype)
            for factor in factor_data
        ]

        # This is tricky.

        # unioned.sum(axis=1) gives us an array of the same size as `dates`
        # containing, for each date, the number of assets that passed our
        # filters on that date.

        # Running this through add.accumulate gives us an array containing, for
        # each date, the running total of the number of assets that passed our
        # filters on or before that date.

        # This means that (bounds[i - 1], bounds[i]) gives us the slice bounds
        # of rows in our output DataFrame corresponding to each date.
        dt_start = 0
        bounds = add.accumulate(unioned.sum(axis=1))
        for dt_idx, dt_end in enumerate(bounds):

            row_bounds = slice(dt_start, dt_end)
            column_indices = nonzero_ys[row_bounds]

            raw_dates_index[row_bounds] = dates[dt_idx]
            raw_assets_index[row_bounds] = assets[column_indices]
            for computed, output in zip(factor_data, factor_outputs):
                output[row_bounds] = computed[dt_idx, column_indices]

            # Upper bound of current row becomes lower bound for next row.
            dt_start = dt_end

        return DataFrame(
            dict(zip(factor_names, factor_outputs)),
            index=MultiIndex.from_arrays(
                [raw_dates_index, raw_assets_index],
            )
        ).tz_localize('UTC', level=0)
