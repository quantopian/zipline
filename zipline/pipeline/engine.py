"""
Compute Engine definitions for the Pipeline API.
"""
from abc import (
    ABCMeta,
    abstractmethod,
)
from uuid import uuid4

from six import (
    iteritems,
    with_metaclass,
)
from six.moves import zip_longest
from numpy import array

from pandas import (
    DataFrame,
    date_range,
    MultiIndex,
)

from zipline.lib.adjusted_array import ensure_ndarray
from zipline.errors import NoFurtherDataError
from zipline.utils.numpy_utils import repeat_first_axis, repeat_last_axis
from zipline.utils.pandas_utils import explode

from .term import AssetExists


class PipelineEngine(with_metaclass(ABCMeta)):

    @abstractmethod
    def run_pipeline(self, pipeline, start_date, end_date):
        """
        Compute values for `pipeline` between `start_date` and `end_date`.

        Returns a DataFrame with a MultiIndex of (date, asset) pairs

        Parameters
        ----------
        pipeline : zipline.pipeline.Pipeline
            The pipeline to run.
        start_date : pd.Timestamp
            Start date of the computed matrix.
        end_date : pd.Timestamp
            End date of the computed matrix.

        Returns
        -------
        result : pd.DataFrame
            A frame of computed results.

            The columns `result` correspond wil be the computed results of
            `pipeline.columns`, which should be a dictionary mapping strings to
            instances of `zipline.pipeline.term.Term`.

            For each date between `start_date` and `end_date`, `result` will
            contain a row for each asset that passed `pipeline.screen`.  A
            screen of None indicates that a row should be returned for each
            asset that existed each day.
        """
        raise NotImplementedError("run_pipeline")


class NoOpPipelineEngine(PipelineEngine):
    """
    A PipelineEngine that doesn't do anything.
    """
    def run_pipeline(self, pipeline, start_date, end_date):
        return DataFrame(
            index=MultiIndex.from_product(
                [date_range(start=start_date, end=end_date, freq='D'), ()],
            ),
            columns=sorted(pipeline.columns.keys()),
        )


class SimplePipelineEngine(object):
    """
    PipelineEngine class that computes each term independently.

    Parameters
    ----------
    loader : PipelineLoader
        A loader to use to retrieve raw data for atomic terms.
    calendar : DatetimeIndex
        Array of dates to consider as trading days when computing a range
        between a fixed start and end.
    asset_finder : zipline.assets.AssetFinder
        An AssetFinder instance.  We depend on the AssetFinder to determine
        which assets are in the top-level universe at any point in time.
    """
    __slots__ = [
        '_loader_dispatch',
        '_calendar',
        '_finder',
        '_root_mask_term',
        '__weakref__',
    ]

    def __init__(self, loader_dispatch, calendar, asset_finder):
        self._loader_dispatch = loader_dispatch
        self._calendar = calendar
        self._finder = asset_finder
        self._root_mask_term = AssetExists()

    def run_pipeline(self, pipeline, start_date, end_date):
        """
        Compute a pipeline.

        Parameters
        ----------
        pipeline : zipline.pipeline.Pipeline
            The pipeline to run.
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

        Step 0 is performed by `zipline.pipeline.graph.TermGraph`.
        Step 1 is performed in `self._compute_root_mask`.
        Step 2 is performed in `self.compute_chunk`.
        Steps 3, 4, and 5 are performed in self._format_factor_matrix.

        See Also
        --------
        PipelineEngine.run_pipeline
        """
        if end_date <= start_date:
            raise ValueError(
                "start_date must be before end_date \n"
                "start_date=%s, end_date=%s" % (start_date, end_date)
            )

        screen_name = uuid4().hex
        graph = pipeline.to_graph(screen_name, self._root_mask_term)
        extra_rows = graph.extra_rows[self._root_mask_term]
        root_mask = self._compute_root_mask(start_date, end_date, extra_rows)
        dates, assets, root_mask_values = explode(root_mask)

        outputs = self.compute_chunk(
            graph,
            dates,
            assets,
            initial_workspace={self._root_mask_term: root_mask_values},
        )

        out_dates = dates[extra_rows:]
        screen_values = outputs.pop(screen_name)

        return self._to_narrow(outputs, screen_values, out_dates, assets)

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
                msg="Insufficient data to compute Pipeline mask: "
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

    def _mask_and_dates_for_atomic_terms(self, terms, workspace, graph, dates):
        max_extra_rows = max(graph.extra_rows[term] for term in terms)

        mask = self._root_mask_term
        offset = graph.extra_rows[mask] - max_extra_rows
        return workspace[mask][offset:], dates[offset:]

    @staticmethod
    def _inputs_for_term(term, workspace, graph):
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

    def _atomic_terms_for_loader(self, graph, loader):
        loader_dispatch = self.loader_dispatch
        for term in graph.atomic_terms:
            if loader_dispatch(term) == loader:
                yield term

    def loader_dispatch(self, term):
        if term is AssetExists():
            return None

        loader = self._loader_dispatch(term)
        if loader is None:
            raise ValueError("Couldn't find loader for %s" % term)
        return loader

    def compute_chunk(self, graph, dates, assets, initial_workspace):
        """
        Compute the Pipeline terms in the graph for the requested start and end
        dates.

        Parameters
        ----------
        graph : zipline.pipeline.graph.TermGraph
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
        loader_dispatch = self.loader_dispatch

        # Copy the supplied initial workspace so we don't mutate it in place.
        workspace = initial_workspace.copy()

        for term in graph.ordered():
            # `term` may have been supplied in `initial_workspace`, and in the
            # future we may pre-compute atomic terms coming from the same
            # dataset.  In either case, we will already have an entry for this
            # term, which we shouldn't re-compute.
            if term in workspace:
                continue

            if term.atomic:
                loader = loader_dispatch(term)
                to_load = sorted(self._atomic_terms_for_loader(graph, loader),
                                 key=lambda t: t.dataset)
                mask, mask_dates = self._mask_and_dates_for_atomic_terms(
                    to_load, workspace, graph, dates,
                )
                loaded = loader.load_adjusted_array(
                    to_load, mask_dates, assets, mask,
                )
                assert len(to_load) == len(loaded)
                for loaded_term, adj_array in zip_longest(to_load, loaded):
                    workspace[loaded_term] = adj_array
            else:
                # Asset labels are always the same, but date labels vary by how
                # many extra rows are needed.
                mask, mask_dates = self._mask_and_dates_for_term(
                    term, workspace, graph, dates
                )
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

    def _to_narrow(self, data, mask, dates, assets):
        """
        Convert raw computed pipeline results into a DataFrame for public APIs.

        Parameters
        ----------
        data : dict[str -> ndarray[ndim=2]]
            Dict mapping column names to computed results.
        mask : ndarray[bool, ndim=2]
            Mask array of values to keep.
        dates : ndarray[datetime64, ndim=1]
            Row index for arrays `data` and `mask`
        assets : ndarray[int64, ndim=2]
            Column index for arrays `data` and `mask`

        Returns
        -------
        results : pd.DataFrame
            The indices of `results` are as follows:

            index : two-tiered MultiIndex of (date, asset).
                Contains an entry for each (date, asset) pair corresponding to
                a `True` value in `mask`.
            columns : Index of str
                One column per entry in `data`.

        If mask[date, asset] is True, then result.loc[(date, asset), colname]
        will contain the value of data[colname][date, asset].
        """
        resolved_assets = array(self._finder.retrieve_all(assets))
        dates_kept = repeat_last_axis(dates.values, len(assets))[mask]
        assets_kept = repeat_first_axis(resolved_assets, len(dates))[mask]
        return DataFrame(
            data={name: arr[mask] for name, arr in iteritems(data)},
            index=MultiIndex.from_arrays([dates_kept, assets_kept]),
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
