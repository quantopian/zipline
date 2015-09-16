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
    itervalues,
    with_metaclass,
)
from six.moves import (
    reduce,
    zip_longest,
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
from zipline.utils.pandas_utils import explode

from .classifier import Classifier
from .factor import Factor
from .filter import Filter
from .graph import TermGraph
from .term import AssetExists


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
        terms : dict[str -> zipline.modelling.term.Term]
            Dict mapping term names to instances.  The supplied names are used
            as column names in our output frame.
        start_date : pd.Timestamp
            Start date of the computed matrix.
        end_date : pd.Timestamp
            End date of the computed matrix.

        Returns
        -------
        matrix : pd.DataFrame
            A matrix of computed results.
        """
        raise NotImplementedError("factor_matrix")


class NoOpFFCEngine(FFCEngine):
    """
    FFCEngine that doesn't do anything.
    """

    def factor_matrix(self, terms, start_date, end_date):
        return DataFrame(
            index=MultiIndex.from_product(
                [date_range(start=start_date, end=end_date, freq='D'), ()],
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
        '_root_mask_term',
        '__weakref__',
    ]

    def __init__(self, loader, calendar, asset_finder):
        self._loader = loader
        self._calendar = calendar
        self._finder = asset_finder
        self._root_mask_term = AssetExists()

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

        Step 0 is performed by `zipline.modelling.graph.TermGraph`.
        Step 1 is performed in `self._compute_root_mask`.
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

        graph = TermGraph(terms)
        extra_rows = graph.extra_rows[self._root_mask_term]

        root_mask = self._compute_root_mask(start_date, end_date, extra_rows)
        dates, assets, root_mask_values = explode(root_mask)
        raw_outputs = self.compute_chunk(
            graph,
            dates,
            assets,
            initial_workspace={self._root_mask_term: root_mask_values},
        )

        # Collect the results that we'll actually show to the user.
        filters, factors = {}, {}
        for name, term in iteritems(terms):
            if isinstance(term, Filter):
                filters[name] = raw_outputs[name]
            elif isinstance(term, Factor):
                factors[name] = raw_outputs[name]
            elif isinstance(term, Classifier):
                continue
            else:
                raise ValueError("Unknown term type: %s" % term)

        # Add the root mask as an implicit filter, truncating off the extra
        # rows that we only needed to compute other terms.
        filters['base'] = root_mask_values[extra_rows:]
        out_dates = dates[extra_rows:]

        return self._format_factor_matrix(out_dates, assets, filters, factors)

    def _compute_root_mask(self, start_date, end_date, extra_rows):
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
            Number of extra rows to compute before `start_date`.
            Extra rows are needed by terms like moving averages that require a
            trailing window of data.

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

        # Build lifetimes matrix reaching back to `extra_rows` days before
        # `start_date.`
        lifetimes = finder.lifetimes(
            calendar[start_idx - extra_rows:end_idx],
            include_start_date=False
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

    def _mask_and_dates_for_term(self, term, workspace, graph, dates):
        """
        Load mask and mask row labels for term.
        """
        mask = term.mask
        offset = graph.extra_rows[mask] - graph.extra_rows[term]
        return workspace[mask][offset:], dates[offset:]

    def _inputs_for_term(self, term, workspace, graph):
        """
        Compute inputs for the given term.

        This is mostly complicated by the fact that for each input we store as
        many rows as will be necessary to serve **any** computation requiring
        that input.
        """
        offsets = graph.offset
        if term.windowed:
            # If term is windowed, then all input data should be instances of
            # AdjustedArray.
            return [
                workspace[input_].traverse(
                    window_length=term.window_length,
                    offset=offsets[term, input_]
                )
                for input_ in term.inputs
            ]

        # If term is not windowed, input_data may be an AdjustedArray or
        # np.ndarray.  Coerce the former to the latter.
        out = []
        for input_ in term.inputs:
            input_data = ensure_ndarray(workspace[input_])
            offset = offsets[term, input_]
            # OPTIMIZATION: Don't make a copy by doing input_data[0:] if
            # offset is zero.
            if offset:
                input_data = input_data[offset:]
            out.append(input_data)
        return out

    def compute_chunk(self, graph, dates, assets, initial_workspace):
        """
        Compute the FFC terms in the graph for the requested start and end
        dates.

        Parameters
        ----------
        graph : zipline.modelling.graph.TermGraph
        dates : pd.DatetimeIndex
            Row labels for our root mask.
        assets : pd.Int64Index
            Column labels for our root mask.
        initial_workspace : dict
            Map from term -> output.
            Must contain at least entry for `self._root_mask_term` whose shape
            is `(len(dates), len(assets))`, but may contain additional
            pre-computed terms for testing or optimization purposes.

        Returns
        -------
        results : dict
            Dictionary mapping requested results to outputs.
        """
        self._validate_compute_chunk_params(dates, assets, initial_workspace)
        loader = self._loader

        # Copy the supplied initial workspace so we don't mutate it in place.
        workspace = initial_workspace.copy()

        for term in graph.ordered():
            # `term` may have been supplied in `initial_workspace`, and in the
            # future we may pre-compute atomic terms coming from the same
            # dataset.  In either case, we will already have an entry for this
            # term, which we shouldn't re-compute.
            if term in workspace:
                continue

            # Asset labels are always the same, but date labels vary by how
            # many extra rows are needed.
            mask, mask_dates = self._mask_and_dates_for_term(
                term, workspace, graph, dates
            )
            if term.atomic:
                # FUTURE OPTIMIZATION: Scan the resolution order for terms in
                # the same dataset and load them here as well.
                to_load = [term]
                loaded = loader.load_adjusted_array(
                    to_load, mask_dates, assets, mask,
                )
                assert len(to_load) == len(loaded)
                for loaded_term, adj_array in zip_longest(to_load, loaded):
                    workspace[loaded_term] = adj_array
            else:
                workspace[term] = term._compute(
                    self._inputs_for_term(term, workspace, graph),
                    mask_dates,
                    assets,
                    mask,
                )
                assert(workspace[term].shape == mask.shape)

        out = {}
        graph_extra_rows = graph.extra_rows
        for name, term in iteritems(graph.outputs):
            # Truncate off extra rows from outputs.
            out[name] = workspace[term][graph_extra_rows[term]:]
        return out

    def _format_factor_matrix(self, dates, assets, filters, factors):
        """
        Convert raw computed filters/factors into a DataFrame for public APIs.

        Parameters
        ----------
        dates : np.array[datetime64]
            Row index for arrays in `filters` and `factors.`
        assets : np.array[int64]
            Column index for arrays in `filters` and `factors.`
        filters : dict
            Dict mapping filter names -> computed filters.
        factors : dict
            Dict mapping factor names -> computed factors.

        Returns
        -------
        factor_matrix : pd.DataFrame
            The indices of `factor_matrix` are as follows:

            index : two-tiered MultiIndex of (date, asset).
                For each date, we return a row for each asset that passed all
                filters on that date.
            columns : keys from `factor_data`

        Each date/asset/factor triple contains the computed value of the given
        factor on the given date for the given asset.
        """
        # FUTURE OPTIMIZATION: Cythonize all of this.

        # Boolean mask of values that passed all filters.
        unioned = reduce(and_, itervalues(filters))

        # Parallel arrays of (x,y) coords for (date, asset) pairs that passed
        # all filters.  Each entry here will correspond to a row in our output
        # frame.
        nonzero_xs, nonzero_ys = unioned.nonzero()

        # Raw arrays storing (date, asset) pairs.
        # These will form the index of our output frame.
        raw_dates_index = empty_like(nonzero_xs, dtype='datetime64[ns]')
        raw_assets_index = empty_like(nonzero_xs, dtype=int)

        # Mapping from column_name -> array.
        # This will be the `data` arg to our output frame.
        columns = {
            name: empty_like(nonzero_xs, dtype=factor.dtype)
            for name, factor in iteritems(factors)
        }
        # We're going to iterate over `iteritems(columns)` a whole bunch of
        # times down below.  It's faster to construct iterate over a tuple of
        # pairs.
        columns_iter = tuple(iteritems(columns))

        # This is tricky.

        # unioned.sum(axis=1) gives us an array of the same size as `dates`
        # containing, for each date, the number of assets that passed our
        # filters on that date.

        # Running this through add.accumulate gives us an array containing, for
        # each date, the running total of the number of assets that passed our
        # filters on or before that date.

        # This means that (bounds[i - 1], bounds[i]) gives us the indices of
        # the first and last rows in our output frame for each date in `dates`.
        bounds = add.accumulate(unioned.sum(axis=1))
        day_start = 0
        for day_idx, day_end in enumerate(bounds):

            day_bounds = slice(day_start, day_end)
            column_indices = nonzero_ys[day_bounds]

            raw_dates_index[day_bounds] = dates[day_idx]
            raw_assets_index[day_bounds] = assets[column_indices]
            for name, colarray in columns_iter:
                colarray[day_bounds] = factors[name][day_idx, column_indices]

            # Upper bound of current row becomes lower bound for next row.
            day_start = day_end

        return DataFrame(
            data=columns,
            index=MultiIndex.from_arrays(
                [
                    raw_dates_index,
                    # FUTURE OPTIMIZATION:
                    # Avoid duplicate lookups by grouping and only looking up
                    # each unique sid once.
                    self._finder.retrieve_all(raw_assets_index),
                ],
            )
        ).tz_localize('UTC', level=0)

    def _validate_compute_chunk_params(self, dates, assets, initial_workspace):
        """
        Verify that the values passed to compute_chunk are well-formed.
        """
        root = self._root_mask_term
        clsname = type(self).__name__
        # Writing this out explicitly so this errors in testing if we change
        # the name without updating this line.
        compute_chunk_name = self.compute_chunk.__name__
        if root not in initial_workspace:
            raise AssertionError(
                "root_mask values not supplied to {cls}.{method}".format(
                    cls=clsname,
                    method=compute_chunk_name,
                )
            )

        shape = initial_workspace[root].shape
        implied_shape = len(dates), len(assets)
        if shape != implied_shape:
            raise AssertionError(
                "root_mask shape is {shape}, but received dates/assets "
                "imply that shape should be {implied}".format(
                    shape=shape,
                    implied=implied_shape,
                )
            )
