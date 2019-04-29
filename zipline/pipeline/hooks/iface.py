from zipline.utils.compat import contextmanager as _contextmanager

from interface import Interface


# Keep track of which methods of PipelineHooks are contextmanagers. Used by
# DelegatingHooks to properly delegate to sub-hooks.
PIPELINE_HOOKS_CONTEXT_MANAGERS = set()


def contextmanager(f):
    """
    Wrapper for contextlib.contextmanager that tracks which methods of
    PipelineHooks are contextmanagers in CONTEXT_MANAGER_METHODS.
    """
    PIPELINE_HOOKS_CONTEXT_MANAGERS.add(f.__name__)
    return _contextmanager(f)


class PipelineHooks(Interface):
    """
    Interface for instrumenting SimplePipelineEngine executions.

    Methods with names like 'on_event()' should be normal methods. They will be
    called by the engine after the corresponding event.

    Methods with names like 'doing_thing()' should be context managers. They
    will be entered by the engine around the corresponding event.

    Methods
    -------
    on_create_execution_plan(self, plan)
    running_pipeline(self, pipeline, start_date, end_date, chunked)
    computing_chunk(self, plan, start_date, end_date)
    loading_terms(self, terms)
    computing_term(self, term):
    """

    def on_create_execution_plan(self, plan):
        """Called on resolution of a Pipeline to an ExecutionPlan.
        """

    @contextmanager
    def running_pipeline(self, pipeline, start_date, end_date):
        """
        Contextmanager entered during execution of run_pipeline or
        run_chunked_pipeline.
        """

    @contextmanager
    def computing_chunk(self, plan, start_date, end_date):
        """Contextmanager entered during execution of compute_chunk.
        """

    @contextmanager
    def loading_terms(self, terms):
        """Contextmanager entered when loading a batch of LoadableTerms.
        """

    @contextmanager
    def computing_term(self, term):
        """Contextmanager entered when computing a ComputableTerm.
        """
