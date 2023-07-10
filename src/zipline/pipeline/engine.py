"""
Computation engines for executing Pipelines.

This module defines the core computation algorithms for executing Pipelines.

The primary entrypoint of this file is SimplePipelineEngine.run_pipeline, which
implements the following algorithm for executing pipelines:

1. Determine the domain of the pipeline. The domain determines the
   top-level set of dates and assets that serve as row- and
   column-labels for the computations performed by this
   pipeline. This logic lives in
   zipline.pipeline.domain.infer_domain.

2. Build a dependency graph of all terms in `pipeline`, with
   information about how many extra rows each term needs from its
   inputs. At this point we also **specialize** any generic
   LoadableTerms to the domain determined in (1). This logic lives in
   zipline.pipeline.graph.TermGraph and
   zipline.pipeline.graph.ExecutionPlan.

3. Combine the domain computed in (2) with our AssetFinder to produce
   a "lifetimes matrix". The lifetimes matrix is a DataFrame of
   booleans whose labels are dates x assets. Each entry corresponds
   to a (date, asset) pair and indicates whether the asset in
   question was tradable on the date in question. This logic
   primarily lives in AssetFinder.lifetimes.

4. Call self._populate_initial_workspace, which produces a
   "workspace" dictionary containing cached or otherwise pre-computed
   terms. By default, the initial workspace contains the lifetimes
   matrix and its date labels.

5. Topologically sort the graph constructed in (1) to produce an
   execution order for any terms that were not pre-populated.  This
   logic lives in TermGraph.

6. Iterate over the terms in the order computed in (5). For each term:

   a. Fetch the term's inputs from the workspace, possibly removing
      unneeded leading rows from the input (see ExecutionPlan.offset
      for details on why we might have extra leading rows).

   b. Call ``term._compute`` with the inputs. Store the results into
      the workspace.

   c. Decrement "reference counts" on the term's inputs, and remove
      their results from the workspace if the refcount hits 0. This
      significantly reduces the maximum amount of memory that we
      consume during execution

   This logic lives in SimplePipelineEngine.compute_chunk.

7. Extract the pipeline's outputs from the workspace and convert them
   into "narrow" format, with output labels dictated by the Pipeline's
   screen. This logic lives in SimplePipelineEngine._to_narrow.
"""
from abc import ABC, abstractmethod
from functools import partial

import pandas as pd
from numpy import arange, array
from toolz import groupby

from zipline.errors import NoFurtherDataError
from zipline.lib.adjusted_array import ensure_adjusted_array, ensure_ndarray
from zipline.utils.date_utils import compute_date_range_chunks
from zipline.utils.input_validation import expect_types
from zipline.utils.numpy_utils import as_column, repeat_first_axis, repeat_last_axis
from zipline.utils.pandas_utils import categorical_df_concat, explode
from zipline.utils.string_formatting import bulleted_list

from .domain import GENERIC, Domain
from .graph import maybe_specialize
from .hooks import DelegatingHooks
from .term import AssetExists, InputDates, LoadableTerm


class PipelineEngine(ABC):
    @abstractmethod
    def run_pipeline(self, pipeline, start_date, end_date, hooks=None):
        """Compute values for ``pipeline`` from ``start_date`` to ``end_date``.

        Parameters
        ----------
        pipeline : zipline.pipeline.Pipeline
            The pipeline to run.
        start_date : pd.Timestamp
            Start date of the computed matrix.
        end_date : pd.Timestamp
            End date of the computed matrix.
        hooks : list[implements(PipelineHooks)], optional
            Hooks for instrumenting Pipeline execution.

        Returns
        -------
        result : pd.DataFrame
            A frame of computed results.

            The ``result`` columns correspond to the entries of
            `pipeline.columns`, which should be a dictionary mapping strings to
            instances of :class:`zipline.pipeline.Term`.

            For each date between ``start_date`` and ``end_date``, ``result``
            will contain a row for each asset that passed `pipeline.screen`.
            A screen of ``None`` indicates that a row should be returned for
            each asset that existed each day.
        """
        raise NotImplementedError("run_pipeline")

    @abstractmethod
    def run_chunked_pipeline(
        self, pipeline, start_date, end_date, chunksize, hooks=None
    ):
        """Compute values for ``pipeline`` from ``start_date`` to ``end_date``, in
        date chunks of size ``chunksize``.

        Chunked execution reduces memory consumption, and may reduce
        computation time depending on the contents of your pipeline.

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
        hooks : list[implements(PipelineHooks)], optional
            Hooks for instrumenting Pipeline execution.

        Returns
        -------
        result : pd.DataFrame
            A frame of computed results.

            The ``result`` columns correspond to the entries of
            `pipeline.columns`, which should be a dictionary mapping strings to
            instances of :class:`zipline.pipeline.Term`.

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
    """Raised if a user tries to call pipeline_output in an algorithm that hasn't
    set up a pipeline engine.
    """


class ExplodingPipelineEngine(PipelineEngine):
    """A PipelineEngine that doesn't do anything."""

    def run_pipeline(self, pipeline, start_date, end_date, hooks=None):
        raise NoEngineRegistered(
            "Attempted to run a pipeline but no pipeline " "resources were registered."
        )

    def run_chunked_pipeline(
        self, pipeline, start_date, end_date, chunksize, hooks=None
    ):
        raise NoEngineRegistered(
            "Attempted to run a chunked pipeline but no pipeline "
            "resources were registered."
        )


def default_populate_initial_workspace(
    initial_workspace, root_mask_term, execution_plan, dates, assets
):
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
    """PipelineEngine class that computes each term independently.

    Parameters
    ----------
    get_loader : callable
        A function that is given a loadable term and returns a PipelineLoader
        to use to retrieve raw data for that term.
    asset_finder : zipline.assets.AssetFinder
        An AssetFinder instance.  We depend on the AssetFinder to determine
        which assets are in the top-level universe at any point in time.
    populate_initial_workspace : callable, optional
        A function which will be used to populate the initial workspace when
        computing a pipeline. See
        :func:`zipline.pipeline.engine.default_populate_initial_workspace`
        for more info.
    default_hooks : list, optional
        List of hooks that should be used to instrument all pipelines executed
        by this engine.

    See Also
    --------
    :func:`zipline.pipeline.engine.default_populate_initial_workspace`
    """

    __slots__ = (
        "_get_loader",
        "_finder",
        "_root_mask_term",
        "_root_mask_dates_term",
        "_populate_initial_workspace",
    )

    @expect_types(
        default_domain=Domain,
        __funcname="SimplePipelineEngine",
    )
    def __init__(
        self,
        get_loader,
        asset_finder,
        default_domain=GENERIC,
        populate_initial_workspace=None,
        default_hooks=None,
    ):
        self._get_loader = get_loader
        self._finder = asset_finder

        self._root_mask_term = AssetExists()
        self._root_mask_dates_term = InputDates()

        self._populate_initial_workspace = (
            populate_initial_workspace or default_populate_initial_workspace
        )
        self._default_domain = default_domain

        if default_hooks is None:
            self._default_hooks = []
        else:
            self._default_hooks = list(default_hooks)

    def run_chunked_pipeline(
        self, pipeline, start_date, end_date, chunksize, hooks=None
    ):
        """Compute values for ``pipeline`` from ``start_date`` to ``end_date``, in
        date chunks of size ``chunksize``.

        Chunked execution reduces memory consumption, and may reduce
        computation time depending on the contents of your pipeline.

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
        hooks : list[implements(PipelineHooks)], optional
            Hooks for instrumenting Pipeline execution.

        Returns
        -------
        result : pd.DataFrame
            A frame of computed results.

            The ``result`` columns correspond to the entries of
            `pipeline.columns`, which should be a dictionary mapping strings to
            instances of :class:`zipline.pipeline.Term`.

            For each date between ``start_date`` and ``end_date``, ``result``
            will contain a row for each asset that passed `pipeline.screen`.
            A screen of ``None`` indicates that a row should be returned for
            each asset that existed each day.

        See Also
        --------
        :meth:`zipline.pipeline.engine.PipelineEngine.run_pipeline`
        """
        domain = self.resolve_domain(pipeline)
        ranges = compute_date_range_chunks(
            domain.sessions(),
            start_date,
            end_date,
            chunksize,
        )
        hooks = self._resolve_hooks(hooks)

        run_pipeline = partial(self._run_pipeline_impl, pipeline, hooks=hooks)
        with hooks.running_pipeline(pipeline, start_date, end_date):
            chunks = [run_pipeline(s, e) for s, e in ranges]

        if len(chunks) == 1:
            # OPTIMIZATION: Don't make an extra copy in `categorical_df_concat`
            # if we don't have to.
            return chunks[0]

        # Filter out empty chunks. Empty dataframes lose dtype information,
        # which makes concatenation fail.
        nonempty_chunks = [c for c in chunks if len(c)]
        return categorical_df_concat(nonempty_chunks, inplace=True)

    def run_pipeline(self, pipeline, start_date, end_date, hooks=None):
        """Compute values for ``pipeline`` from ``start_date`` to ``end_date``.

        Parameters
        ----------
        pipeline : zipline.pipeline.Pipeline
            The pipeline to run.
        start_date : pd.Timestamp
            Start date of the computed matrix.
        end_date : pd.Timestamp
            End date of the computed matrix.
        hooks : list[implements(PipelineHooks)], optional
            Hooks for instrumenting Pipeline execution.

        Returns
        -------
        result : pd.DataFrame
            A frame of computed results.

            The ``result`` columns correspond to the entries of
            `pipeline.columns`, which should be a dictionary mapping strings to
            instances of :class:`zipline.pipeline.Term`.

            For each date between ``start_date`` and ``end_date``, ``result``
            will contain a row for each asset that passed `pipeline.screen`.
            A screen of ``None`` indicates that a row should be returned for
            each asset that existed each day.
        """
        hooks = self._resolve_hooks(hooks)
        with hooks.running_pipeline(pipeline, start_date, end_date):
            return self._run_pipeline_impl(
                pipeline,
                start_date,
                end_date,
                hooks,
            )

    def _run_pipeline_impl(self, pipeline, start_date, end_date, hooks):
        """Shared core for ``run_pipeline`` and ``run_chunked_pipeline``."""
        # See notes at the top of this module for a description of the
        # algorithm implemented here.
        if end_date < start_date:
            raise ValueError(
                "start_date must be before or equal to end_date \n"
                f"start_date={start_date}, end_date={end_date}"
            )

        domain = self.resolve_domain(pipeline)

        plan = pipeline.to_execution_plan(
            domain,
            self._root_mask_term,
            start_date,
            end_date,
        )
        extra_rows = plan.extra_rows[self._root_mask_term]
        root_mask = self._compute_root_mask(
            domain,
            start_date,
            end_date,
            extra_rows,
        )
        dates, sids, root_mask_values = explode(root_mask)

        workspace = self._populate_initial_workspace(
            {
                self._root_mask_term: root_mask_values,
                self._root_mask_dates_term: as_column(dates.values),
            },
            self._root_mask_term,
            plan,
            dates,
            sids,
        )

        refcounts = plan.initial_refcounts(workspace)
        execution_order = plan.execution_order(workspace, refcounts)

        with hooks.computing_chunk(execution_order, start_date, end_date):
            results = self.compute_chunk(
                graph=plan,
                dates=dates,
                sids=sids,
                workspace=workspace,
                refcounts=refcounts,
                execution_order=execution_order,
                hooks=hooks,
            )

        return self._to_narrow(
            plan.outputs,
            results,
            results.pop(plan.screen_name),
            dates[extra_rows:],
            sids,
        )

    def _compute_root_mask(self, domain, start_date, end_date, extra_rows):
        """Compute a lifetimes matrix from our AssetFinder, then drop columns that
        didn't exist at all during the query dates.

        Parameters
        ----------
        domain : zipline.pipeline.domain.Domain
            Domain for which we're computing a pipeline.
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
        sessions = domain.sessions()

        if start_date not in sessions:
            raise ValueError(
                f"Pipeline start date ({start_date}) is not a trading session for "
                f"domain {domain}."
            )

        elif end_date not in sessions:
            raise ValueError(
                f"Pipeline end date {end_date} is not a trading session for "
                f"domain {domain}."
            )

        start_idx, end_idx = sessions.slice_locs(start_date, end_date)
        if start_idx < extra_rows:
            raise NoFurtherDataError.from_lookback_window(
                initial_message="Insufficient data to compute Pipeline:",
                first_date=sessions[0],
                lookback_start=start_date,
                lookback_length=extra_rows,
            )

        # NOTE: This logic should probably be delegated to the domain once we
        #       start adding more complex domains.
        #
        # Build lifetimes matrix reaching back to `extra_rows` days before
        # `start_date.`
        finder = self._finder
        lifetimes = finder.lifetimes(
            sessions[start_idx - extra_rows : end_idx],
            include_start_date=False,
            country_codes=(domain.country_code,),
        )

        if not lifetimes.columns.unique:
            columns = lifetimes.columns
            duplicated = columns[columns.duplicated()].unique()
            raise AssertionError("Duplicated sids: %d" % duplicated)

        # Filter out columns that didn't exist from the farthest look back
        # window through the end of the requested dates.
        existed = lifetimes.any()
        ret = lifetimes.loc[:, existed]
        num_assets = ret.shape[1]

        if num_assets == 0:
            raise ValueError(
                "Failed to find any assets with country_code {!r} that traded "
                "between {} and {}.\n"
                "This probably means that your asset db is old or that it has "
                "incorrect country/exchange metadata.".format(
                    domain.country_code,
                    start_date,
                    end_date,
                )
            )

        return ret

    @staticmethod
    def _inputs_for_term(term, workspace, graph, domain, refcounts):
        """
        Compute inputs for the given term.

        This is mostly complicated by the fact that for each input we store as
        many rows as will be necessary to serve **any** computation requiring
        that input.
        """
        offsets = graph.offset
        out = []

        # We need to specialize here because we don't change ComputableTerm
        # after resolving domains, so they can still contain generic terms as
        # inputs.
        specialized = [maybe_specialize(t, domain) for t in term.inputs]

        if term.windowed:
            # If term is windowed, then all input data should be instances of
            # AdjustedArray.
            for input_ in specialized:
                adjusted_array = ensure_adjusted_array(
                    workspace[input_],
                    input_.missing_value,
                )
                out.append(
                    adjusted_array.traverse(
                        window_length=term.window_length,
                        offset=offsets[term, input_],
                        # If the refcount for the input is > 1, we will need
                        # to traverse this array again so we must copy.
                        # If the refcount for the input == 0, this is the last
                        # traversal that will happen so we can invalidate
                        # the AdjustedArray and mutate the data in place.
                        copy=refcounts[input_] > 1,
                    )
                )
        else:
            # If term is not windowed, input_data may be an AdjustedArray or
            # np.ndarray. Coerce the former to the latter.
            for input_ in specialized:
                input_data = ensure_ndarray(workspace[input_])
                offset = offsets[term, input_]
                input_data = input_data[offset:]
                if refcounts[input_] > 1:
                    input_data = input_data.copy()
                out.append(input_data)
        return out

    def compute_chunk(
        self, graph, dates, sids, workspace, refcounts, execution_order, hooks
    ):
        """Compute the Pipeline terms in the graph for the requested start and end
        dates.

        This is where we do the actual work of running a pipeline.

        Parameters
        ----------
        graph : zipline.pipeline.graph.ExecutionPlan
            Dependency graph of the terms to be executed.
        dates : pd.DatetimeIndex
            Row labels for our root mask.
        sids : pd.Int64Index
            Column labels for our root mask.
        workspace : dict
            Map from term -> output.
            Must contain at least entry for `self._root_mask_term` whose shape
            is `(len(dates), len(assets))`, but may contain additional
            pre-computed terms for testing or optimization purposes.
        refcounts : dict[Term, int]
            Dictionary mapping terms to number of dependent terms. When a
            term's refcount hits 0, it can be safely discarded from
            ``workspace``. See TermGraph.decref_dependencies for more info.
        execution_order : list[Term]
            Order in which to execute terms.
        hooks : implements(PipelineHooks)
            Hooks to instrument pipeline execution.

        Returns
        -------
        results : dict
            Dictionary mapping requested results to outputs.
        """
        self._validate_compute_chunk_params(graph, dates, sids, workspace)

        get_loader = self._get_loader

        # Copy the supplied initial workspace so we don't mutate it in place.
        workspace = workspace.copy()
        domain = graph.domain

        # Many loaders can fetch data more efficiently if we ask them to
        # retrieve all their inputs at once. For example, a loader backed by a
        # SQL database can fetch multiple columns from the database in a single
        # query.
        #
        # To enable these loaders to fetch their data efficiently, we group
        # together requests for LoadableTerms if they are provided by the same
        # loader and they require the same number of extra rows.
        #
        # The extra rows condition is a simplification: we don't currently have
        # a mechanism for asking a loader to fetch different windows of data
        # for different terms, so we only batch requests together when they're
        # going to produce data for the same set of dates.
        def loader_group_key(term):
            loader = get_loader(term)
            extra_rows = graph.extra_rows[term]
            return loader, extra_rows

        # Only produce loader groups for the terms we expect to load.  This
        # ensures that we can run pipelines for graphs where we don't have a
        # loader registered for an atomic term if all the dependencies of that
        # term were supplied in the initial workspace.
        will_be_loaded = graph.loadable_terms - workspace.keys()
        loader_groups = groupby(
            loader_group_key,
            (t for t in execution_order if t in will_be_loaded),
        )

        for term in execution_order:
            # `term` may have been supplied in `initial_workspace`, or we may
            # have loaded `term` as part of a batch with another term coming
            # from the same loader (see note on loader_group_key above). In
            # either case, we already have the term computed, so don't
            # recompute.
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
                loader = get_loader(term)
                to_load = sorted(
                    loader_groups[loader_group_key(term)], key=lambda t: t.dataset
                )
                self._ensure_can_load(loader, to_load)
                with hooks.loading_terms(to_load):
                    loaded = loader.load_adjusted_array(
                        domain,
                        to_load,
                        mask_dates,
                        sids,
                        mask,
                    )
                assert set(loaded) == set(to_load), (
                    "loader did not return an AdjustedArray for each column\n"
                    "expected: %r\n"
                    "got:      %r"
                    % (
                        sorted(to_load, key=repr),
                        sorted(loaded, key=repr),
                    )
                )
                workspace.update(loaded)
            else:
                with hooks.computing_term(term):
                    workspace[term] = term._compute(
                        self._inputs_for_term(
                            term,
                            workspace,
                            graph,
                            domain,
                            refcounts,
                        ),
                        mask_dates,
                        sids,
                        mask,
                    )
                if term.ndim == 2:
                    assert workspace[term].shape == mask.shape
                else:
                    assert workspace[term].shape == (mask.shape[0], 1)

                # Decref dependencies of ``term``, and clear any terms
                # whose refcounts hit 0.
                for garbage in graph.decref_dependencies(term, refcounts):
                    del workspace[garbage]

        # At this point, all the output terms are in the workspace.
        out = {}
        graph_extra_rows = graph.extra_rows
        for name, term in graph.outputs.items():
            # Truncate off extra rows from outputs.
            out[name] = workspace[term][graph_extra_rows[term] :]
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
            return pd.DataFrame(
                data={name: array([], dtype=arr.dtype) for name, arr in data.items()},
                index=pd.MultiIndex.from_arrays([empty_dates, empty_assets]),
            )
        # if "open_instance" in data.keys():
        #     data["open_instance"].tofile("../../open_instance.dat")
        final_columns = {}
        for name in data:
            # Each term that computed an output has its postprocess method
            # called on the filtered result.
            #
            # Using this to convert np.records to tuples
            final_columns[name] = terms[name].postprocess(data[name][mask])

        resolved_assets = array(self._finder.retrieve_all(assets))
        index = _pipeline_output_index(dates, resolved_assets, mask)
        return pd.DataFrame(
            data=final_columns, index=index, columns=final_columns.keys()
        )

    def _validate_compute_chunk_params(self, graph, dates, sids, initial_workspace):
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
        implied_shape = len(dates), len(sids)
        if shape != implied_shape:
            raise AssertionError(
                "root_mask shape is {shape}, but received dates/assets "
                "imply that shape should be {implied}".format(
                    shape=shape,
                    implied=implied_shape,
                )
            )

        for term in initial_workspace:
            if self._is_special_root_term(term):
                continue

            if term.domain is GENERIC:
                # XXX: We really shouldn't allow **any** generic terms to be
                # populated in the initial workspace. A generic term, by
                # definition, can't correspond to concrete data until it's
                # paired with a domain, and populate_initial_workspace isn't
                # given the domain of execution, so it can't possibly know what
                # data to use when populating a generic term.
                #
                # In our current implementation, however, we don't have a good
                # way to represent specializations of ComputableTerms that take
                # only generic inputs, so there's no good way for the initial
                # workspace to provide data for such terms except by populating
                # the generic ComputableTerm.
                #
                # The right fix for the above is to implement "full
                # specialization", i.e., implementing ``specialize`` uniformly
                # across all terms, not just LoadableTerms. Having full
                # specialization will also remove the need for all of the
                # remaining ``maybe_specialize`` calls floating around in this
                # file.
                #
                # In the meantime, disallowing ComputableTerms in the initial
                # workspace would break almost every test in
                # `test_filter`/`test_factor`/`test_classifier`, and fixing
                # them would require updating all those tests to compute with
                # more specialized terms. Once we have full specialization, we
                # can fix all the tests without a large volume of edits by
                # simply specializing their workspaces, so for now I'm leaving
                # this in place as a somewhat sharp edge.
                if isinstance(term, LoadableTerm):
                    raise ValueError(
                        "Loadable workspace terms must be specialized to a "
                        "domain, but got generic term {}".format(term)
                    )

            elif term.domain != graph.domain:
                raise ValueError(
                    "Initial workspace term {} has domain {}. "
                    "Does not match pipeline domain {}".format(
                        term,
                        term.domain,
                        graph.domain,
                    )
                )

    def resolve_domain(self, pipeline):
        """Resolve a concrete domain for ``pipeline``."""
        domain = pipeline.domain(default=self._default_domain)
        if domain is GENERIC:
            raise ValueError(
                "Unable to determine domain for Pipeline.\n"
                "Pass domain=<desired domain> to your Pipeline to set a "
                "domain."
            )
        return domain

    def _is_special_root_term(self, term):
        return term is self._root_mask_term or term is self._root_mask_dates_term

    def _resolve_hooks(self, hooks):
        if hooks is None:
            hooks = []
        return DelegatingHooks(self._default_hooks + hooks)

    def _ensure_can_load(self, loader, terms):
        """Ensure that ``loader`` can load ``terms``."""
        if not loader.currency_aware:
            bad = [t for t in terms if t.currency_conversion is not None]
            if bad:
                raise ValueError(
                    "Requested currency conversion is not supported for the "
                    "following terms:\n{}".format(bulleted_list(bad))
                )


def _pipeline_output_index(dates, assets, mask):
    """
    Create a MultiIndex for a pipeline output.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Row labels for ``mask``.
    assets : pd.Index
        Column labels for ``mask``.
    mask : np.ndarray[bool]
        Mask array indicating date/asset pairs that should be included in
        output index.

    Returns
    -------
    index : pd.MultiIndex
        MultiIndex  containing (date,  asset) pairs  corresponding to  ``True``
        values in ``mask``.
    """
    date_labels = repeat_last_axis(arange(len(dates)), len(assets))[mask]
    asset_labels = repeat_first_axis(arange(len(assets)), len(dates))[mask]
    return pd.MultiIndex(
        [dates, assets],
        [date_labels, asset_labels],
        # TODO: We should probably add names for these.
        names=[None, None],
        verify_integrity=False,
    )
