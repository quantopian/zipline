import itertools
from operator import attrgetter

import numpy as np
import pandas as pd
import pytest
import toolz
from numpy.testing import assert_almost_equal

from zipline.pipeline import Pipeline
from zipline.pipeline.classifiers import Everything
from zipline.pipeline.data import Column, DataSet
from zipline.pipeline.data.testing import TestingDataSet
from zipline.pipeline.domain import US_EQUITIES
from zipline.pipeline.factors import CustomFactor
from zipline.pipeline.hooks.progress import (
    ProgressHooks,
    repr_htmlsafe,
    TestingProgressPublisher,
)
from zipline.pipeline.hooks.testing import TestingHooks
from zipline.pipeline.term import AssetExists, ComputableTerm, LoadableTerm
from zipline.testing import parameter_space
from zipline.testing.fixtures import (
    ZiplineTestCase,
    WithSeededRandomPipelineEngine,
)
from zipline.testing.predicates import instance_of


class TrivialFactor(CustomFactor):
    """
    A CustomFactor that doesn't do any work.

    This is used to test that we correctly track date bounds in hooks in the
    presence of windowed computations.
    """

    window_length = 10
    inputs = [TestingDataSet.float_col, TestingDataSet.datetime_col]

    def compute(self, today, assets, out, *inputs):
        pass


class HooksTestCase(WithSeededRandomPipelineEngine, ZiplineTestCase):
    """Tests for verifying that SimplePipelineEngine calls hooks as expected."""

    ASSET_FINDER_COUNTRY_CODE = "US"

    @classmethod
    def make_seeded_random_pipeline_engine_hooks(cls):
        # Inject a testing hook as a default hook to verify that default hooks
        # are invoked properly.
        cls.global_testing_hook = TestingHooks()
        return [cls.global_testing_hook]

    def init_instance_fixtures(self):
        super(HooksTestCase, self).init_instance_fixtures()
        # Clear out the global testing hook after each test run.
        self.add_instance_callback(self.global_testing_hook.clear)

    @parameter_space(
        nhooks=[0, 1, 2],
        chunked=[True, False],
    )
    def test_engine_calls_hooks(self, nhooks, chunked):
        # Pass multiple hooks to make sure we call methods on all of them.
        hooks = [TestingHooks() for _ in range(nhooks)]

        pipeline = Pipeline(
            {
                "bool_": TestingDataSet.bool_col.latest,
                "factor_rank": TrivialFactor().rank().zscore(),
            },
            domain=US_EQUITIES,
        )
        start_date, end_date = self.trading_days[[-10, -1]]

        if chunked:
            self.run_chunked_pipeline(
                pipeline=pipeline,
                start_date=start_date,
                end_date=end_date,
                chunksize=5,
                hooks=hooks,
            )
            expected_chunks = [
                tuple(self.trading_days[[-10, -6]]),
                tuple(self.trading_days[[-5, -1]]),
            ]
        else:
            self.run_pipeline(
                pipeline=pipeline,
                start_date=start_date,
                end_date=end_date,
                hooks=hooks,
            )
            expected_chunks = [(start_date, end_date)]

        expected_loads = set(TrivialFactor.inputs) | {TestingDataSet.bool_col}
        expected_computes = {
            TestingDataSet.bool_col.latest,
            TrivialFactor(),
            TrivialFactor().rank(),
            TrivialFactor().rank().zscore(),
            Everything(),  # Default input for .rank().
        }

        for h in hooks + [self.global_testing_hook]:
            self.verify_trace(
                h.trace,
                pipeline,
                pipeline_start_date=start_date,
                pipeline_end_date=end_date,
                expected_loads=expected_loads,
                expected_computes=expected_computes,
                expected_chunks=expected_chunks,
            )

    def verify_trace(
        self,
        trace,
        pipeline,
        pipeline_start_date,
        pipeline_end_date,
        expected_loads,
        expected_computes,
        expected_chunks,
    ):
        """Verify a trace of a Pipeline execution."""
        # First/last calls should bracket the pipeline execution.
        self.expect_context_pair(trace[0], trace[-1], "running_pipeline")
        assert trace[0].args == (pipeline, pipeline_start_date, pipeline_end_date)

        # Break up the trace into the traces of each chunk.
        chunk_traces = self.split_by_chunk(trace[1:-1])

        for ctrace, (chunk_start, chunk_end) in zip(chunk_traces, expected_chunks):
            # Next call should bracket compute_chunk
            self.expect_context_pair(ctrace[0], ctrace[-1], "computing_chunk")
            assert isinstance(ctrace[0].args[0], list)  # terms
            assert ctrace[0].args[1:] == (chunk_start, chunk_end)

            # Remainder of calls should be loads and computes. These have to
            # happen in dependency order, but we don't bother to assert that
            # here. We just make sure that we see each expected load/compute
            # exactly once.
            loads_and_computes = ctrace[1:-1]
            loads = set()
            computes = set()
            for enter, exit_ in two_at_a_time(loads_and_computes):
                self.expect_context_pair(enter, exit_, method=None)

                if enter.method_name == "loading_terms":
                    for loaded_term in enter.args[0]:
                        # We should only see each term once.
                        assert loaded_term not in loads
                        # Don't worry about domains here.
                        loads.add(loaded_term.unspecialize())
                elif enter.method_name == "computing_term":
                    computed_term = enter.args[0]
                    assert computed_term not in computes
                    computes.add(computed_term)
                else:
                    raise ValueError("Unexpected method: {}".format(enter.method_name))

            assert loads == expected_loads
            assert computes == expected_computes

    def split_by_chunk(self, trace):
        """
        Split a trace of a chunked pipeline execution into a list of traces for
        each chunk.
        """

        def is_end_of_chunk(call):
            return call.method_name == "computing_chunk" and call.state == "exit"

        to_yield = []
        for call in trace:
            to_yield.append(call)
            if is_end_of_chunk(call):
                yield to_yield
                to_yield = []

        # Make sure all calls were part of a chunk.
        assert to_yield == []

    def expect_context_pair(self, enter, exit_, method):
        assert enter.state == "enter"
        assert exit_.state == "exit"

        if method is None:
            # Just assert that the methods match.
            assert enter.call is exit_.call
        else:
            assert enter.call.method_name == method


class ShouldGetSkipped(DataSet):
    """
    Dataset that's only used by PrepopulatedFactor. It should get pruned from
    the execution when PrepopulatedFactor is prepopulated.
    """

    column1 = Column(dtype=float)
    column2 = Column(dtype=float)


class PrepopulatedFactor(CustomFactor):
    """CustomFactor that will be set by populate_initial_workspace."""

    window_length = 5
    inputs = [ShouldGetSkipped.column1, ShouldGetSkipped.column2]

    def compute(self, today, assets, out, col1, col2):
        out[:] = 0.0


PREPOPULATED_TERM = PrepopulatedFactor()


class ProgressHooksTestCase(WithSeededRandomPipelineEngine, ZiplineTestCase):
    """Tests for verifying ProgressHooks."""

    ASSET_FINDER_COUNTRY_CODE = "US"

    START_DATE = pd.Timestamp("2014-01-02")
    END_DATE = pd.Timestamp("2014-01-31")

    # Don't populate PREPOPULATED_TERM for days after this cutoff.
    # This is used to test that we correctly compute progress when the number
    # of terms computed in each chunk changes.
    PREPOPULATED_TERM_CUTOFF = END_DATE - pd.Timedelta("2 days")

    @classmethod
    def make_seeded_random_populate_initial_workspace(cls):
        # Populate valeus for PREPOPULATED_TERM. This is used to ensure that we
        # properly track progress when we skip prepopulated terms.
        def populate(initial_workspace, root_mask_term, execution_plan, dates, assets):
            if PREPOPULATED_TERM not in execution_plan:
                return initial_workspace
            elif dates[-1] > cls.PREPOPULATED_TERM_CUTOFF:
                return initial_workspace

            workspace = initial_workspace.copy()
            _, dates = execution_plan.mask_and_dates_for_term(
                PREPOPULATED_TERM,
                root_mask_term,
                workspace,
                dates,
            )
            shape = (len(dates), len(assets))
            workspace[PREPOPULATED_TERM] = np.zeros(shape, dtype=float)
            return workspace

        return populate

    @classmethod
    def make_seeded_random_loader_columns(cls):
        return TestingDataSet.columns | ShouldGetSkipped.columns

    def test_progress_hooks(self):
        publisher = TestingProgressPublisher()
        hooks = [ProgressHooks.with_static_publisher(publisher)]
        pipeline = Pipeline(
            {
                "bool_": TestingDataSet.bool_col.latest,
                "factor_rank": TrivialFactor().rank().zscore(),
                "prepopulated": PREPOPULATED_TERM,
            },
            domain=US_EQUITIES,
        )
        start_date, end_date = self.trading_days[[-10, -1]]
        expected_chunks = [
            tuple(self.trading_days[[-10, -6]]),
            tuple(self.trading_days[[-5, -1]]),
        ]

        # First chunk should get prepopulated term in initial workspace.
        assert expected_chunks[0][1] < self.PREPOPULATED_TERM_CUTOFF

        # Second chunk should have to compute PREPOPULATED_TERM explicitly.
        assert expected_chunks[0][1] < self.PREPOPULATED_TERM_CUTOFF

        self.run_chunked_pipeline(
            pipeline=pipeline,
            start_date=start_date,
            end_date=end_date,
            chunksize=5,
            hooks=hooks,
        )

        self.verify_trace(
            publisher.trace,
            pipeline_start_date=start_date,
            pipeline_end_date=end_date,
            expected_chunks=expected_chunks,
        )

    def test_progress_hooks_empty_pipeline(self):
        publisher = TestingProgressPublisher()
        hooks = [ProgressHooks.with_static_publisher(publisher)]
        pipeline = Pipeline({}, domain=US_EQUITIES)
        start_date, end_date = self.trading_days[[-10, -1]]
        expected_chunks = [
            tuple(self.trading_days[[-10, -6]]),
            tuple(self.trading_days[[-5, -1]]),
        ]

        self.run_chunked_pipeline(
            pipeline=pipeline,
            start_date=start_date,
            end_date=end_date,
            chunksize=5,
            hooks=hooks,
        )

        self.verify_trace(
            publisher.trace,
            pipeline_start_date=start_date,
            pipeline_end_date=end_date,
            expected_chunks=expected_chunks,
            empty=True,
        )

    def verify_trace(
        self,
        trace,
        pipeline_start_date,
        pipeline_end_date,
        expected_chunks,
        empty=False,
    ):
        # Percent complete should be monotonically increasing through the whole
        # execution.
        for before, after in toolz.sliding_window(2, trace):
            assert after.percent_complete >= before.percent_complete

        # First publish should come from the start of the first chunk, with no
        # work yet.
        first = trace[0]
        expected_first = TestingProgressPublisher.TraceState(
            state="init",
            percent_complete=0.0,
            execution_bounds=(pipeline_start_date, pipeline_end_date),
            current_chunk_bounds=expected_chunks[0],
            current_work=None,
        )
        assert first == expected_first

        # Last publish should have a state of success and be 100% complete.
        last = trace[-1]
        expected_last = TestingProgressPublisher.TraceState(
            state="success",
            percent_complete=100.0,
            execution_bounds=(pipeline_start_date, pipeline_end_date),
            current_chunk_bounds=expected_chunks[-1],
            # We don't know what the last work item will be, but it must be an
            # instance of a single ComputableTerm, because we only run
            # ComputableTerms one at a time, and a LoadableTerm will only be in
            # the graph if some ComputableTerm depends on it.
            #
            # The one exception to this rule is that, if we run a completely
            # empty pipeline, the final work will be None.
            current_work=None if empty else [instance_of(ComputableTerm)],
        )
        assert last == expected_last

        # Remaining updates should all be loads or computes.
        middle = trace[1:-1]
        for update in middle:
            # For empty pipelines we never leave the 'init' state.
            if empty:
                assert update.state == "init"
                assert update.current_work is None
                continue

            if update.state in ("loading", "computing"):
                assert isinstance(update.current_work, list)
            if update.state == "loading":
                for term in update.current_work:
                    assert isinstance(term, (LoadableTerm, AssetExists))
            elif update.state == "computing":
                for term in update.current_work:
                    assert isinstance(term, ComputableTerm)
            else:
                raise AssertionError(
                    "Unexpected state: {}".format(update.state),
                )

        # Break up the remaining updates by chunk.
        all_chunks = []
        grouped = itertools.groupby(middle, attrgetter("current_chunk_bounds"))
        for (chunk_start, chunk_stop), chunk_trace in grouped:
            all_chunks.append((chunk_start, chunk_stop))

            chunk_trace = list(chunk_trace)
            expected_end_progress = self.expected_chunk_progress(
                pipeline_start_date,
                pipeline_end_date,
                chunk_stop,
            )
            end_progress = chunk_trace[-1].percent_complete
            assert_almost_equal(
                end_progress,
                expected_end_progress,
            )

        assert all_chunks == expected_chunks

    @parameter_space(chunked=[True, False])
    def test_error_handling(self, chunked):
        publisher = TestingProgressPublisher()
        hooks = [ProgressHooks.with_static_publisher(publisher)]

        class SomeError(Exception):
            pass

        class ExplodingFactor(CustomFactor):
            inputs = [TestingDataSet.float_col]
            window_length = 1

            def compute(self, *args, **kwargs):
                raise SomeError()

        pipeline = Pipeline({"boom": ExplodingFactor()}, domain=US_EQUITIES)
        start_date, end_date = self.trading_days[[-10, -1]]

        with pytest.raises(SomeError):
            if chunked:
                self.run_chunked_pipeline(
                    pipeline=pipeline,
                    start_date=start_date,
                    end_date=end_date,
                    chunksize=5,
                    hooks=hooks,
                )
            else:
                self.run_pipeline(
                    pipeline=pipeline,
                    start_date=start_date,
                    end_date=end_date,
                    hooks=hooks,
                )

        final_update = publisher.trace[-1]
        assert final_update.state == "error"

    def expected_chunk_progress(self, pipeline_start, pipeline_end, chunk_end):
        """Get expected progress after finishing a chunk ending at ``chunk_end``."""
        # +1 to be inclusive of end dates
        total_days = (pipeline_end - pipeline_start).days + 1
        days_complete = (chunk_end - pipeline_start).days + 1
        return round((100.0 * days_complete) / total_days, 3)


class TestTermRepr:
    def test_htmlsafe_repr(self):
        class MyFactor(CustomFactor):
            inputs = [TestingDataSet.float_col]
            window_length = 3

        assert repr_htmlsafe(MyFactor()) == repr(MyFactor())

    def test_htmlsafe_repr_escapes_html(self):
        class MyFactor(CustomFactor):
            inputs = [TestingDataSet.float_col]
            window_length = 3

            def __repr__(self):
                return "<b>foo</b>"

        assert repr_htmlsafe(MyFactor()) == "<b>foo</b>".replace("<", "&lt;").replace(
            ">", "&gt;"
        )

    def test_htmlsafe_repr_handles_errors(self):
        class MyFactor(CustomFactor):
            inputs = [TestingDataSet.float_col]
            window_length = 3

            def __repr__(self):
                raise ValueError("Kaboom!")

        assert repr_htmlsafe(MyFactor()) == "(Error Displaying MyFactor)"

    def test_htmlsafe_repr_escapes_html_when_it_handles_errors(self):
        class MyFactor(CustomFactor):
            inputs = [TestingDataSet.float_col]
            window_length = 3

            def __repr__(self):
                raise ValueError("Kaboom!")

        MyFactor.__name__ = "<b>foo</b>"
        converted = MyFactor.__name__.replace("<", "&lt;").replace(">", "&gt;")

        assert repr_htmlsafe(MyFactor()) == "(Error Displaying {})".format(converted)


def two_at_a_time(it):
    """Iterate over ``it``, two elements at a time.

    ``it`` must yield an even number of times.

    Examples
    --------
    >>> list(two_at_a_time([1, 2, 3, 4]))
    [(1, 2), (3, 4)]
    """
    return toolz.partition(2, it, pad=None)
