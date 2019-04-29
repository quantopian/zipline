"""Pipeline hooks for tracking and displaying progress.
"""
import cgi
from collections import namedtuple
import time

from interface import implements

from zipline.utils.compat import contextmanager
from zipline.utils.string_formatting import bulleted_list

from .iface import PipelineHooks


class ProgressHooks(implements(PipelineHooks)):
    """
    Hooks implementation for displaying progress.

    Parameters
    ----------
    publisher_factory : callable
        Function producing a new object with a ``publish()`` method that takes
        a ``ProgressModel`` and publishes progress to a consumer.
    """
    def __init__(self, publisher_factory):
        # This object encapsulates state that should be reset between pipeline
        # executions.
        self._state = _ProgressHooksState(publisher_factory)

    @classmethod
    def with_widget_publisher(cls):
        """
        Construct a ProgressHooks that publishes to Jupyter via
        ``IPython.display``.
        """
        return cls(publisher_factory=IPythonWidgetProgressPublisher)

    @classmethod
    def with_static_publisher(cls, publisher):
        """Construct a ProgressHooks that uses an already-constructed publisher.
        """
        return cls(publisher_factory=lambda: publisher)

    # The state manages our ProgressModel because we receive the information
    # necessary to initialize over the course of several hook callbacks.
    @property
    def _model(self):
        return self._state.model

    def _publish(self):
        self._state.publish()

    @contextmanager
    def running_pipeline(self, pipeline, start_date, end_date):
        self._state.set_bounds(start_date, end_date)
        try:
            yield
        except Exception:
            self._model.finish(success=False)
            self._publish()
            raise
        else:
            self._model.finish(success=True)
            self._publish()
        finally:
            self._state.reset()

    def on_create_execution_plan(self, plan):
        self._state.set_execution_plan(plan)
        self._publish()

    @contextmanager
    def computing_chunk(self, plan, start_date, end_date):
        try:
            self._model.start_chunk(start_date, end_date)
            self._publish()
            yield
        finally:
            self._model.finish_chunk(start_date, end_date)
            self._publish()

    @contextmanager
    def loading_terms(self, terms):
        try:
            self._model.start_load_terms(terms)
            self._publish()
            yield
        finally:
            self._model.finish_load_terms(terms)
            self._publish()

    @contextmanager
    def computing_term(self, term):
        try:
            self._model.start_compute_term(term)
            self._publish()
            yield
        finally:
            self._model.finish_compute_term(term)
            self._publish()


class _ProgressHooksState(object):
    """
    Helper class for managing incremental acquisition of pipeline state in
    ``ProgressHooks``.
    """

    def __init__(self, publisher_factory):
        self._publisher_factory = publisher_factory
        self.reset()

    def reset(self):
        self._start_date = None
        self._end_date = None
        self._plan = None

        self._model = None
        self._publisher = None

    def set_bounds(self, start_date, end_date):
        self._start_date = start_date
        self._end_date = end_date

    def set_execution_plan(self, plan):
        # NOTE: This will be called multiple times in a chunked execution, but
        # the plans should always be functionally equivalent.
        self._plan = plan

    def publish(self):
        self.publisher.publish(self.model)

    @property
    def model(self):
        _model = self._model
        if _model is None:
            _model = self._model = self._create_model()
        return _model

    @property
    def publisher(self):
        publisher = self._publisher
        if publisher is None:
            publisher = self._publisher = self._publisher_factory()
        return self._publisher

    def _create_model(self):
        """
        Create a ProgressModel.

        Can only be called after ``set_bounds`` and ``set_execution_plan``.
        """
        start_date = self._start_date
        end_date = self._end_date
        plan = self._plan

        if start_date is None:
            raise ValueError(
                "Must have start_date to create a progress model."
            )
        elif end_date is None:
            raise ValueError(
                "Must have end_date to create a progress model."
            )
        elif plan is None:
            raise ValueError(
                "Must have an execution plan to create a progress model."
            )

        # Subtract one from the execution plan length to account for
        # AssetExists(), which will always be in the pipeline, but won't be
        # computed.
        return ProgressModel(len(plan) - 1, start_date, end_date)


class ProgressModel(object):
    """
    Model object for tracking progress of a Pipeline execution.

    Parameters
    ----------
    nterms : int
        Number of terms in the execution plan of the Pipeline being run.
    start_date : pd.Timestamp
        Start date of the range over which ``plan`` will be computed.
    end_date : pd.Timestamp
        End date of the range over which ``plan`` will be computed.

    Methods
    -------
    start_chunk(start_date, end_date)
    finish_chunk(start_date, end_date)
    start_load_terms(terms)
    finish_load_terms(terms)
    start_compute_term(term)
    finish_compute_term(term)
    finish(success)

    Attributes
    ----------
    state : {'init', 'loading', 'computing', 'error', 'success'}
        Current state of the execution.
    percent_complete : float
        Percent of execution that has been completed, on a scale from 0 to 100.
    execution_time : float
        Number of seconds that the execution required. Only available if state
        is 'error' or 'success'.
    execution_bounds : (pd.Timestamp, pd.Timestamp)
        Pair of (start_date, end_date) for the entire execution.
    current_chunk_bounds : (pd.Timestamp, pd.Timestamp)
        Pair of (start_date, end_date) for the currently executing chunk.
    current_work : [zipline.pipeline.term.Term]
        List of terms currently being loaded or computed.
    """

    def __init__(self, nterms, start_date, end_date):
        self._start_date = start_date
        self._end_date = end_date

        # +1 to be inclusive of end_date.
        total_days = (end_date - start_date).days + 1
        self._max_progress = total_days * nterms

        self._progress = 0
        self._days_completed = 0

        self._state = 'init'

        self._current_chunk_size = None
        self._current_chunk_bounds = None
        self._current_work = None

        self._start_time = time.time()
        self._end_time = None

    # These properties form the public interface for Publishers.
    @property
    def state(self):
        return self._state

    @property
    def percent_complete(self):
        return 100.0 * float(self._progress) / self._max_progress

    @property
    def execution_time(self):
        if self._end_time is None:
            raise ValueError(
                "Can't get execution_time until execution is complete."
            )
        return self._end_time - self._start_time

    @property
    def execution_bounds(self):
        return (self._start_date, self._end_date)

    @property
    def current_chunk_bounds(self):
        return self._current_chunk_bounds

    @property
    def current_work(self):
        return self._current_work

    # These methods form the interface for ProgressHooks.
    def start_chunk(self, start_date, end_date):
        days_since_start = (end_date - self._start_date).days + 1
        self._current_chunk_size = days_since_start - self._days_completed
        self._current_chunk_bounds = (start_date, end_date)

    def finish_chunk(self, start_date, end_date):
        self._days_completed += self._current_chunk_size

    def start_load_terms(self, terms):
        self._state = 'loading'
        self._current_work = terms

    def finish_load_terms(self, terms):
        self._increment_progress(nterms=len(terms))

    def start_compute_term(self, term):
        self._state = 'computing'
        self._current_work = [term]

    def finish_compute_term(self, term):
        self._increment_progress(nterms=1)

    def finish(self, success):
        self._end_time = time.time()
        if success:
            self._state = 'success'
        else:
            self._state = 'error'

    def _increment_progress(self, nterms):
        self._progress += nterms * self._current_chunk_size


try:
    import ipywidgets
    HAVE_WIDGETS = True
except ImportError:
    HAVE_WIDGETS = False

try:
    from IPython.display import display, HTML as IPython_HTML
    HAVE_IPYTHON = True
except ImportError:
    HAVE_IPYTHON = False


# XXX: This class is currently untested, because we don't require ipywidgets as
#      a test dependency. Be careful if you make changes to this.
class IPythonWidgetProgressPublisher(object):
    """A progress publisher that publishes to an IPython/Jupyter widget.
    """

    def __init__(self):
        missing = []
        if not HAVE_WIDGETS:
            missing.append('ipywidgets')
        elif not HAVE_IPYTHON:
            missing.append('IPython')

        if missing:
            raise ValueError(
                "IPythonWidgetProgressPublisher needs ipywidgets and IPython:"
                "\nMissing:\n{}".format(bulleted_list(missing))
            )

        # Heading for progress display.
        self._heading = ipywidgets.HTML()

        # Percent Complete Indicator to the left of the bar.
        indicator_width = '120px'
        self._percent_indicator = ipywidgets.HTML(
            layout={'width': indicator_width},
        )

        # The progress bar itself.
        self._bar = ipywidgets.FloatProgress(
            value=0.0,
            min=0.0,
            max=100.0,
            bar_style='info',
            # Leave enough space for the percent indicator.
            layout={'width': 'calc(100% - {})'.format(indicator_width)},
        )
        bar_and_percent = ipywidgets.HBox([self._percent_indicator, self._bar])

        # Collapsable details tab underneath the progress bar.
        self._details_body = ipywidgets.HTML()
        self._details_tab = ipywidgets.Accordion(
            children=[self._details_body],
            selected_index=None,  # Start in collapsed state.
            layout={
                # Override default border settings to make details tab less
                # heavy.
                'border': '1px',
            },
        )
        # There's no public interface for setting title in the constructor :/.
        self._details_tab.set_title(0, 'Details')

        # Container for the combined widget.
        self._layout = ipywidgets.VBox(
            [
                self._heading,
                bar_and_percent,
                self._details_tab,
            ],
            # Overall layout consumes 75% of the page.
            layout={'width': '75%'},
        )

        self._displayed = False

    def publish(self, model):
        if model.state == 'init':
            self._heading.value = '<b>Analyzing Pipeline...</b>'
            self._set_progress(0.0)

            if not self._displayed:
                display(self._layout)
                self._displayed = True

        elif model.state in ('loading', 'computing'):
            term_list = self._render_term_list(model.current_work)
            if model.state == 'loading':
                details_heading = '<b>Loading Inputs:</b>'
            else:
                details_heading = '<b>Computing Expression:</b>'

            self._details_body.value = details_heading + term_list
            self._heading.value = (
                "<b>Running Pipeline</b>: Chunk Start={}, Chunk End={}"
                .format(*model.current_chunk_bounds)
            )
            self._set_progress(model.percent_complete)

        elif model.state == 'success':
            # Replace widget layout with html that can be persisted.
            self._layout.close()
            display(
                IPython_HTML("<b>Pipeline Execution Time:</b> {}".format(
                    self._format_execution_time(model.execution_time)
                ))
            )

        elif model.state == 'error':
            self._bar.bar_style = 'danger'
            self._layout.close()

        else:
            raise ValueError('Unknown display state: {!r}'.format(model.state))

    @staticmethod
    def _render_term_list(terms):
        list_elements = ''.join([
             '<li><pre>{}</pre></li>'.format(cgi.escape(str(t)))
             for t in terms
        ])
        return '<ul>{}</ul>'.format(list_elements)

    def _set_progress(self, percent_complete):
        self._bar.value = percent_complete
        self._percent_indicator.value = (
            "<b>{:.2f}% Complete</b>".format(percent_complete)
        )

    @staticmethod
    def _format_execution_time(total_seconds):
        """Helper method for displaying total execution time of a Pipeline.

        Parameters
        ----------
        total_seconds : float
            Number of seconds elapsed.

        Returns
        -------
        formatted : str
            User-facing text representation of elapsed time.
        """
        def maybe_s(n):
            if n == 1:
                return ''
            return 's'

        minutes, seconds = divmod(total_seconds, 60)
        minutes = int(minutes)
        if minutes >= 60:
            hours, minutes = divmod(minutes, 60)
            t = "{hours} Hour{hs}, {minutes} Minute{ms}, {seconds:.2f} Seconds"
            return t.format(
                hours=hours, hs=maybe_s(hours),
                minutes=minutes, ms=maybe_s(minutes),
                seconds=seconds,
            )
        elif minutes >= 1:
            t = "{minutes} Minute{ms}, {seconds:.2f} Seconds"
            return t.format(
                minutes=minutes,
                ms=maybe_s(minutes),
                seconds=seconds,
            )
        else:
            return "{seconds:.2f} Seconds".format(seconds=seconds)


class TestingProgressPublisher(object):
    """A progress publisher that records a trace of model states for testing.
    """
    TraceState = namedtuple('TraceState', [
        'state',
        'percent_complete',
        'execution_bounds',
        'current_chunk_bounds',
        'current_work',
    ])

    def __init__(self):
        self.trace = []

    def publish(self, model):
        self.trace.append(
            self.TraceState(
                state=model.state,
                percent_complete=model.percent_complete,
                execution_bounds=model.execution_bounds,
                current_chunk_bounds=model.current_chunk_bounds,
                current_work=model.current_work
            ),
        )
