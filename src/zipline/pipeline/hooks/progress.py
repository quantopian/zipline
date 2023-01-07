"""Pipeline hooks for tracking and displaying progress.
"""
from collections import namedtuple
import time

from interface import implements

from zipline.utils.compat import contextmanager, escape_html
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
        self._publisher_factory = publisher_factory
        self._reset_transient_state()

    def _reset_transient_state(self):
        self._start_date = None
        self._end_date = None
        self._model = None
        self._publisher = None

    @classmethod
    def with_widget_publisher(cls):
        """
        Construct a ProgressHooks that publishes to Jupyter via
        ``IPython.display``.
        """
        return cls(publisher_factory=IPythonWidgetProgressPublisher)

    @classmethod
    def with_static_publisher(cls, publisher):
        """Construct a ProgressHooks that uses an already-constructed publisher."""
        return cls(publisher_factory=lambda: publisher)

    def _publish(self):
        self._publisher.publish(self._model)

    @contextmanager
    def running_pipeline(self, pipeline, start_date, end_date):
        self._start_date = start_date
        self._end_date = end_date

        try:
            yield
        except Exception:
            if self._model is None:
                # This will only happen if an error happens in the Pipeline
                # Engine beteween entering `running_pipeline` and the first
                # `computing_chunk` call. If that happens, just propagate the
                # exception.
                raise
            self._model.finish(success=False)
            self._publish()
            raise
        else:
            self._model.finish(success=True)
            self._publish()
        finally:
            self._reset_transient_state()

    @contextmanager
    def computing_chunk(self, terms, start_date, end_date):
        # Set up model on first compute_chunk call.
        if self._model is None:
            self._publisher = self._publisher_factory()
            self._model = ProgressModel(
                start_date=self._start_date,
                end_date=self._end_date,
            )

        try:
            self._model.start_chunk(terms, start_date, end_date)
            self._publish()
            yield
        finally:
            self._model.finish_chunk(terms, start_date, end_date)
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


class ProgressModel:
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
    load_precomputed_terms(terms)
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
    current_work : [zipline.pipeline.Term]
        List of terms currently being loaded or computed.
    """

    def __init__(self, start_date, end_date):
        self._start_date = start_date
        self._end_date = end_date

        # +1 to be inclusive of end_date.
        self._total_days = (end_date - start_date).days + 1
        self._progress = 0.0
        self._days_completed = 0

        self._state = "init"

        # Number of days in current chunk.
        self._current_chunk_size = None

        # (start_date, end_date) of current chunk.
        self._current_chunk_bounds = None

        # How much should we increment progress by after completing a term?
        self._completed_term_increment = None

        # How much should we increment progress by after completing a chunk?
        # This is zero unless we compute a pipeline with no terms, in which
        # case it will be the full chunk percentage.
        self._completed_chunk_increment = None

        # Terms currently being computed.
        self._current_work = None

        # Tracking state for total elapsed time.
        self._start_time = time.time()
        self._end_time = None

    # These properties form the interface for Publishers.
    @property
    def state(self):
        return self._state

    @property
    def percent_complete(self):
        return round(self._progress * 100.0, 3)

    @property
    def execution_time(self):
        if self._end_time is None:
            raise ValueError("Can't get execution_time until execution is complete.")
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
    def start_chunk(self, terms, start_date, end_date):
        days_since_start = (end_date - self._start_date).days + 1
        self._current_chunk_size = days_since_start - self._days_completed
        self._current_chunk_bounds = (start_date, end_date)

        # What percent of our overall progress will happen in this chunk?
        chunk_percent = float(self._current_chunk_size) / self._total_days

        # How much of that is associated with each completed term?
        nterms = len(terms)
        if nterms:
            self._completed_term_increment = chunk_percent / len(terms)
            self._completed_chunk_increment = 0.0
        else:
            # Special case. If we don't have any terms, increment the entire
            # chunk's worth of progress when we finish the chunk.
            self._completed_term_increment = 0.0
            self._completed_chunk_increment = chunk_percent

    def finish_chunk(self, terms, start_date, end_date):
        self._days_completed += self._current_chunk_size
        self._progress += self._completed_chunk_increment

    def start_load_terms(self, terms):
        self._state = "loading"
        self._current_work = terms

    def finish_load_terms(self, terms):
        self._finish_terms(nterms=len(terms))

    def start_compute_term(self, term):
        self._state = "computing"
        self._current_work = [term]

    def finish_compute_term(self, term):
        self._finish_terms(nterms=1)

    def finish(self, success):
        self._end_time = time.time()
        if success:
            self._state = "success"
        else:
            self._state = "error"

    def _finish_terms(self, nterms):
        self._progress += nterms * self._completed_term_increment


try:
    import ipywidgets

    HAVE_WIDGETS = True

    # This VBox subclass exists to work around a strange display issue but
    # where the repr of the progress bar sometimes gets re-displayed upon
    # re-opening the notebook, even after the bar has closed. The repr of VBox
    # is somewhat noisy, so we replace it here with a version that just returns
    # an empty string.
    class ProgressBarContainer(ipywidgets.VBox):
        def __repr__(self):
            return ""

except ImportError:
    HAVE_WIDGETS = False

try:
    from IPython.display import display, HTML as IPython_HTML

    HAVE_IPYTHON = True
except ImportError:
    HAVE_IPYTHON = False


# XXX: This class is currently untested, because we don't require ipywidgets as
#      a test dependency. Be careful if you make changes to this.
class IPythonWidgetProgressPublisher:
    """A progress publisher that publishes to an IPython/Jupyter widget."""

    def __init__(self):
        missing = []
        if not HAVE_WIDGETS:
            missing.append("ipywidgets")
        elif not HAVE_IPYTHON:
            missing.append("IPython")

        if missing:
            raise ValueError(
                "IPythonWidgetProgressPublisher needs ipywidgets and IPython:"
                "\nMissing:\n{}".format(bulleted_list(missing))
            )

        # Heading for progress display.
        self._heading = ipywidgets.HTML()

        # Percent Complete Indicator to the left of the bar.
        indicator_width = "120px"
        self._percent_indicator = ipywidgets.HTML(
            layout={"width": indicator_width},
        )

        # The progress bar itself.
        self._bar = ipywidgets.FloatProgress(
            value=0.0,
            min=0.0,
            max=100.0,
            bar_style="info",
            # Leave enough space for the percent indicator.
            layout={"width": "calc(100% - {})".format(indicator_width)},
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
                "border": "1px",
            },
        )
        # There's no public interface for setting title in the constructor :/.
        self._details_tab.set_title(0, "Details")

        # Container for the combined widget.
        self._layout = ProgressBarContainer(
            [
                self._heading,
                bar_and_percent,
                self._details_tab,
            ],
            # Overall layout consumes 75% of the page.
            layout={"width": "75%"},
        )

        self._displayed = False

    def publish(self, model):
        if model.state == "init":
            self._heading.value = "<b>Analyzing Pipeline...</b>"
            self._set_progress(0.0)
            self._ensure_displayed()

        elif model.state in ("loading", "computing"):

            term_list = self._render_term_list(model.current_work)
            if model.state == "loading":
                details_heading = "<b>Loading Inputs:</b>"
            else:
                details_heading = "<b>Computing Expression:</b>"
            self._details_body.value = details_heading + term_list

            chunk_start, chunk_end = model.current_chunk_bounds
            self._heading.value = (
                "<b>Running Pipeline</b>: Chunk Start={}, Chunk End={}".format(
                    chunk_start.date(), chunk_end.date()
                )
            )

            self._set_progress(model.percent_complete)

            self._ensure_displayed()

        elif model.state == "success":
            # Replace widget layout with html that can be persisted.
            self._stop_displaying()
            display(
                IPython_HTML(
                    "<b>Pipeline Execution Time:</b> {}".format(
                        self._format_execution_time(model.execution_time)
                    )
                ),
            )

        elif model.state == "error":
            self._bar.bar_style = "danger"
            self._stop_displaying()
        else:
            self._layout.close()
            raise ValueError("Unknown display state: {!r}".format(model.state))

    def _ensure_displayed(self):
        if not self._displayed:
            display(self._layout)
            self._displayed = True

    def _stop_displaying(self):
        self._layout.close()

    @staticmethod
    def _render_term_list(terms):
        list_elements = "".join(
            ["<li><pre>{}</pre></li>".format(repr_htmlsafe(t)) for t in terms]
        )
        return "<ul>{}</ul>".format(list_elements)

    def _set_progress(self, percent_complete):
        self._bar.value = percent_complete
        self._percent_indicator.value = "<b>{:.2f}% Complete</b>".format(
            percent_complete
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
                return ""
            return "s"

        minutes, seconds = divmod(total_seconds, 60)
        minutes = int(minutes)
        if minutes >= 60:
            hours, minutes = divmod(minutes, 60)
            t = "{hours} Hour{hs}, {minutes} Minute{ms}, {seconds:.2f} Seconds"
            return t.format(
                hours=hours,
                hs=maybe_s(hours),
                minutes=minutes,
                ms=maybe_s(minutes),
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


class TestingProgressPublisher:
    """A progress publisher that records a trace of model states for testing."""

    TraceState = namedtuple(
        "TraceState",
        [
            "state",
            "percent_complete",
            "execution_bounds",
            "current_chunk_bounds",
            "current_work",
        ],
    )

    def __init__(self):
        self.trace = []

    def publish(self, model):
        self.trace.append(
            self.TraceState(
                state=model.state,
                percent_complete=model.percent_complete,
                execution_bounds=model.execution_bounds,
                current_chunk_bounds=model.current_chunk_bounds,
                current_work=model.current_work,
            ),
        )


def repr_htmlsafe(t):
    """Repr a value and html-escape the result.

    If an error is thrown by the repr, show a placeholder.
    """
    try:
        r = repr(t)
    except Exception:
        r = "(Error Displaying {})".format(type(t).__name__)

    return escape_html(str(r), quote=True)
