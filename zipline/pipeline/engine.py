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
from numpy import array
from pandas import DataFrame, MultiIndex
from toolz import groupby, juxt
from toolz.curried.operator import getitem

from zipline.lib.adjusted_array import ensure_adjusted_array, ensure_ndarray
from zipline.errors import NoFurtherDataError
from zipline.utils.numpy_utils import (
    as_column,
    repeat_first_axis,
    repeat_last_axis,
)
from zipline.utils.pandas_utils import explode

from .term import AssetExists, InputDates, LoadableTerm

from zipline.utils.date_utils import compute_date_range_chunks
from zipline.utils.pandas_utils import categorical_df_concat
from zipline.utils.sharedoc import copydoc


class PipelineEngine(with_metaclass(ABCMeta)):

    @abstractmethod
    def run_pipeline(self, pipeline, start_date, end_date):
        """
        Compute values for ``pipeline`` between ``start_date`` and
        ``end_date``.

        Returns a DataFrame with a MultiIndex of (date, asset) pairs.

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

            The ``result`` columns correspond to the entries of
            `pipeline.columns`, which should be a dictionary mapping strings to
            instances of :class:`zipline.pipeline.term.Term`.

            For each date between ``start_date`` and ``end_date``, ``result``
            will contain a row for each asset that passed `pipeline.screen`.
            A screen of ``None`` indicates that a row should be returned for
            each asset that existed each day.
        """
        raise NotImplementedError("run_pipeline")

    @abstractmethod
    def run_chunked_pipeline(self, pipeline, start_date, end_date, chunksize):
        """
        Compute values for `pipeline` in number of days equal to `chunksize`
        and return stitched up result. Computing in chunks is useful for
        pipelines computed over a long period of time.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to run.
        start_date : pd.Timestamp
            The start date to run the pipeline for.
        end_date : pd.Timestamp
            The end date to run the pipeline for.
        chunksize : int
            The number of days to execute at a time.

        Returns
        -------
        result : pd.DataFrame
            A frame of computed results.

            The ``result`` columns correspond to the entries of
            `pipeline.columns`, which should be a dictionary mapping strings to
            instances of :class:`zipline.pipeline.term.Term`.

            For each date between ``start_date`` and ``end_date``, ``result``
            will contain a row for each asset that passed `pipeline.screen`.
            A screen of ``None`` indicates that a row should be returned for
            each asset that existed each day.

        See Also
        --------
        :meth:`zipline.pipeline.engine.PipelineEngine.run_pipeline`
        """
        raise NotImplementedError("run_chunked_pipeline")


class NoEngineRegistered(Exception):
    """
    Raised if a user tries to call pipeline_output in an algorithm that hasn't
    set up a pipeline engine.
    """


class ExplodingPipelineEngine(PipelineEngine):
    """
    A PipelineEngine that doesn't do anything.
    """
    def run_pipeline(self, pipeline, start_date, end_date):
        raise NoEngineRegistered(
            "Attempted to run a pipeline but no pipeline "
            "resources were registered."
        )

    def run_chunked_pipeline(self, pipeline, start_date, end_date, chunksize):
        raise NoEngineRegistered(
            "Attempted to run a chunked pipeline but no pipeline "
            "resources were registered."
        )


def default_populate_initial_workspace(initial_workspace,
                                       root_mask_term,
                                       execution_plan,
                                       dates,
                                       assets):
    """The default implementation for ``populate_initial_workspace``. This
    function returns the ``initial_workspace`` argument without making any
    modifications.

    Parameters
    ----------
    initial_workspace : dict[array-like]
        The initial workspace before we have populated it with any cached
        terms.
    root_mask_term : Term
        The root mask term, normally ``AssetExists()``. This is needed to
        compute the dates for individual terms.
    execution_plan : ExecutionPlan
        The execution plan for the pipeline being run.
    dates : pd.DatetimeIndex
        All of the dates being requested in this pipeline run including
        the extra dates for look back windows.
    assets : pd.Int64Index
        All of the assets that exist for the window being computed.

    Returns
    -------
    populated_initial_workspace : dict[term, array-like]
        The workspace to begin computations with.
    """
    return initial_workspace


class SimplePipelineEngine(PipelineEngine):
    """
    PipelineEngine class that computes each term independently.

    Parameters
    ----------
    get_loader : callable
        A function that is given a loadable term and returns a PipelineLoader
        to use to retrieve raw data for that term.
    calendar : DatetimeIndex
        Array of dates to consider as trading days when computing a range
        between a fixed start and end.
    asset_finder : zipline.assets.AssetFinder
        An AssetFinder instance.  We depend on the AssetFinder to determine
        which assets are in the top-level universe at any point in time.
    populate_initial_workspace : callable, optional
        A function which will be used to populate the initial workspace when
        computing a pipeline. See
        :func:`zipline.pipeline.engine.default_populate_initial_workspace`
        for more info.

    See Also
    --------
    :func:`zipline.pipeline.engine.default_populate_initial_workspace`
    """
    __slots__ = (
        '_get_loader',
        '_calendar',
        '_finder',
        '_root_mask_term',
        '_root_mask_dates_term',
        '_populate_initial_workspace',
    )

    def __init__(self,
                 get_loader,
                 calendar,
                 asset_finder,
                 populate_initial_workspace=None):
        self._get_loader = get_loader
        self._calendar = calendar
        self._finder = asset_finder

        self._root_mask_term = AssetExists()
        self._root_mask_dates_term = InputDates()

        self._populate_initial_workspace = (
            populate_initial_workspace or default_populate_initial_workspace
        )

    def run_pipeline(self, pipeline, start_date, end_date):
        """
        Compute a pipeline.

        The algorithm implemented here can be broken down into the following
        stages:

        0. Build a dependency graph of all terms in `pipeline`.  Topologically
           sort the graph to determine an order in which we can compute the
           terms.

        1. Ask our AssetFinder for a "lifetimes matrix", which should contain,
           for each date between start_date and end_date, a boolean value for
           each known asset indicating whether the asset existed on that date.

        2. Compute each term in the dependency order determined in (0), caching
           the results in a a dictionary to that they can be fed into future
           terms.

        3. For each date, determine the number of assets passing
           pipeline.screen.  The sum, N, of all these values is the total
           number of rows in our output frame, so we pre-allocate an output
           array of length N for each factor in `terms`.

        4. Fill in the arrays allocated in (3) by copying computed values from
           our output cache into the corresponding rows.

        5. Stick the values computed in (4) into a DataFrame and return it.

        Step 0 is performed by ``Pipeline.to_graph``.
        Step 1 is performed in ``SimplePipelineEngine._compute_root_mask``.
        Step 2 is performed in ``SimplePipelineEngine.compute_chunk``.
        Steps 3, 4, and 5 are performed in ``SimplePiplineEngine._to_narrow``.

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

            The ``result`` columns correspond to the entries of
            `pipeline.columns`, which should be a dictionary mapping strings to
            instances of :class:`zipline.pipeline.term.Term`.

            For each date between ``start_date`` and ``end_date``, ``result``
            will contain a row for each asset that passed `pipeline.screen`.
            A screen of ``None`` indicates that a row should be returned for
            each asset that existed each day.

        See Also
        --------
        :meth:`zipline.pipeline.engine.PipelineEngine.run_pipeline`
        :meth:`zipline.pipeline.engine.PipelineEngine.run_chunked_pipeline`
        """
        if end_date < start_date:
            raise ValueError(
                "start_date must be before or equal to end_date \n"
                "start_date=%s, end_date=%s" % (start_date, end_date)
            )

        screen_name = uuid4().hex
        graph = pipeline.to_execution_plan(
            screen_name,
            self._root_mask_term,
            self._calendar,
            start_date,
            end_date,
        )
        extra_rows = graph.extra_rows[self._root_mask_term]
        root_mask = self._compute_root_mask(start_date, end_date, extra_rows)
        dates, assets, root_mask_values = explode(root_mask)

        initial_workspace = self._populate_initial_workspace(
            {
                self._root_mask_term: root_mask_values,
                self._root_mask_dates_term: as_column(dates.values)
            },
            self._root_mask_term,
            graph,
            dates,
            assets,
        )

        results = self.compute_chunk(
            graph,
            dates,
            assets,
            initial_workspace,
        )

        return self._to_narrow(
            graph.outputs,
            results,
            results.pop(screen_name),
            dates[extra_rows:],
            assets,
        )

    @copydoc(PipelineEngine.run_chunked_pipeline)
    def run_chunked_pipeline(self, pipeline, start_date, end_date, chunksize):
        ranges = compute_date_range_chunks(
            self._calendar,
            start_date,
            end_date,
            chunksize,
        )
        chunks = [self.run_pipeline(pipeline, s, e) for s, e in ranges]

        if len(chunks) == 1:
            # OPTIMIZATION: Don't make an extra copy in `categorical_df_concat`
            # if we don't have to.
            return chunks[0]

        return categorical_df_concat(chunks, inplace=True)

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
            raise NoFurtherDataError.from_lookback_window(
                initial_message="Insufficient data to compute Pipeline:",
                first_date=calendar[0],
                lookback_start=start_date,
                lookback_length=extra_rows,
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
        ret = lifetimes.loc[:, existed]
        shape = ret.shape
        assert shape[0] * shape[1] != 0, 'root mask cannot be empty'
        return ret

    @staticmethod
    def _inputs_for_term(term, workspace, graph):
        """
        Compute inputs for the given term.

        This is mostly complicated by the fact that for each input we store as
        many rows as will be necessary to serve **any** computation requiring
        that input.
        """
        offsets = graph.offset
        out = []
        if term.windowed:
            # If term is windowed, then all input data should be instances of
            # AdjustedArray.
            for input_ in term.inputs:
                adjusted_array = ensure_adjusted_array(
                    workspace[input_], input_.missing_value,
                )
                out.append(
                    adjusted_array.traverse(
                        window_length=term.window_length,
                        offset=offsets[term, input_],
                    )
                )
        else:
            # If term is not windowed, input_data may be an AdjustedArray or
            # np.ndarray.  Coerce the former to the latter.
            for input_ in term.inputs:
                input_data = ensure_ndarray(workspace[input_])
                offset = offsets[term, input_]
                # OPTIMIZATION: Don't make a copy by doing input_data[0:] if
                # offset is zero.
                if offset:
                    input_data = input_data[offset:]
                out.append(input_data)
        return out

    def get_loader(self, term):
        return self._get_loader(term)

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
        get_loader = self.get_loader

        # Copy the supplied initial workspace so we don't mutate it in place.
        workspace = initial_workspace.copy()

        # If loadable terms share the same loader and extra_rows, load them all
        # together.
        loader_group_key = juxt(get_loader, getitem(graph.extra_rows))
        loader_groups = groupby(loader_group_key, graph.loadable_terms)

        refcounts = graph.initial_refcounts(workspace)

        for term in graph.execution_order(refcounts):
            # `term` may have been supplied in `initial_workspace`, and in the
            # future we may pre-compute loadable terms coming from the same
            # dataset.  In either case, we will already have an entry for this
            # term, which we shouldn't re-compute.
            if term in workspace:
                continue

            # Asset labels are always the same, but date labels vary by how
            # many extra rows are needed.
            mask, mask_dates = graph.mask_and_dates_for_term(
                term,
                self._root_mask_term,
                workspace,
                dates,
            )

            if isinstance(term, LoadableTerm):
                to_load = sorted(
                    loader_groups[loader_group_key(term)],
                    key=lambda t: t.dataset
                )
                loader = get_loader(term)
                loaded = loader.load_adjusted_array(
                    to_load, mask_dates, assets, mask,
                )
                workspace.update(loaded)
            else:
                workspace[term] = term._compute(
                    self._inputs_for_term(term, workspace, graph),
                    mask_dates,
                    assets,
                    mask,
                )
                if term.ndim == 2:
                    assert workspace[term].shape == mask.shape
                else:
                    assert workspace[term].shape == (mask.shape[0], 1)

                # Decref dependencies of ``term``, and clear any terms whose
                # refcounts hit 0.
                for garbage_term in graph.decref_dependencies(term, refcounts):
                    del workspace[garbage_term]

        out = {}
        graph_extra_rows = graph.extra_rows
        for name, term in iteritems(graph.outputs):
            # Truncate off extra rows from outputs.
            out[name] = workspace[term][graph_extra_rows[term]:]
        return out

    def _to_narrow(self, terms, data, mask, dates, assets):
        """
        Convert raw computed pipeline results into a DataFrame for public APIs.

        Parameters
        ----------
        terms : dict[str -> Term]
            Dict mapping column names to terms.
        data : dict[str -> ndarray[ndim=2]]
            Dict mapping column names to computed results for those names.
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
        if not mask.any():
            # Manually handle the empty DataFrame case. This is a workaround
            # to pandas failing to tz_localize an empty dataframe with a
            # MultiIndex. It also saves us the work of applying a known-empty
            # mask to each array.
            #
            # Slicing `dates` here to preserve pandas metadata.
            empty_dates = dates[:0]
            empty_assets = array([], dtype=object)
            return DataFrame(
                data={
                    name: array([], dtype=arr.dtype)
                    for name, arr in iteritems(data)
                },
                index=MultiIndex.from_arrays([empty_dates, empty_assets]),
            )

        resolved_assets = array(self._finder.retrieve_all(assets))
        dates_kept = repeat_last_axis(dates.values, len(assets))[mask]
        assets_kept = repeat_first_axis(resolved_assets, len(dates))[mask]

        final_columns = {}
        for name in data:
            # Each term that computed an output has its postprocess method
            # called on the filtered result.
            #
            # As of Mon May 2 15:38:47 2016, we only use this to convert
            # LabelArrays into categoricals.
            final_columns[name] = terms[name].postprocess(data[name][mask])

        return DataFrame(
            data=final_columns,
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
