from zipline.errors import UnsupportedPipelineOutput
from zipline.utils.input_validation import (
    expect_element,
    expect_types,
    optional,
)

from .domain import Domain, GENERIC, infer_domain
from .graph import ExecutionPlan, TermGraph, SCREEN_NAME
from .filters import Filter
from .term import AssetExists, ComputableTerm, Term


class Pipeline:
    """
    A Pipeline object represents a collection of named expressions to be
    compiled and executed by a PipelineEngine.

    A Pipeline has two important attributes: 'columns', a dictionary of named
    :class:`~zipline.pipeline.Term` instances, and 'screen', a
    :class:`~zipline.pipeline.Filter` representing criteria for
    including an asset in the results of a Pipeline.

    To compute a pipeline in the context of a TradingAlgorithm, users must call
    ``attach_pipeline`` in their ``initialize`` function to register that the
    pipeline should be computed each trading day. The most recent outputs of an
    attached pipeline can be retrieved by calling ``pipeline_output`` from
    ``handle_data``, ``before_trading_start``, or a scheduled function.

    Parameters
    ----------
    columns : dict, optional
        Initial columns.
    screen : zipline.pipeline.Filter, optional
        Initial screen.
    """

    __slots__ = ("_columns", "_screen", "_domain", "__weakref__")

    @expect_types(columns=optional(dict), screen=optional(Filter), domain=Domain)
    def __init__(self, columns=None, screen=None, domain=GENERIC):
        if columns is None:
            columns = {}

        validate_column = self.validate_column
        for column_name, term in columns.items():
            validate_column(column_name, term)
            if not isinstance(term, ComputableTerm):
                raise TypeError(
                    "Column {column_name!r} contains an invalid pipeline term "
                    "({term}). Did you mean to append '.latest'?".format(
                        column_name=column_name,
                        term=term,
                    )
                )

        self._columns = columns
        self._screen = screen
        self._domain = domain

    @property
    def columns(self):
        """The output columns of this pipeline.

        Returns
        -------
        columns : dict[str, zipline.pipeline.ComputableTerm]
            Map from column name to expression computing that column's output.
        """
        return self._columns

    @property
    def screen(self):
        """
        The screen of this pipeline.

        Returns
        -------
        screen : zipline.pipeline.Filter or None
            Term defining the screen for this pipeline. If ``screen`` is a
            filter, rows that do not pass the filter (i.e., rows for which the
            filter computed ``False``) will be dropped from the output of this
            pipeline before returning results.

        Notes
        -----
        Setting a screen on a Pipeline does not change the values produced for
        any rows: it only affects whether a given row is returned. Computing a
        pipeline with a screen is logically equivalent to computing the
        pipeline without the screen and then, as a post-processing-step,
        filtering out any rows for which the screen computed ``False``.
        """
        return self._screen

    @expect_types(term=Term, name=str)
    def add(self, term, name, overwrite=False):
        """Add a column.

        The results of computing ``term`` will show up as a column in the
        DataFrame produced by running this pipeline.

        Parameters
        ----------
        column : zipline.pipeline.Term
            A Filter, Factor, or Classifier to add to the pipeline.
        name : str
            Name of the column to add.
        overwrite : bool
            Whether to overwrite the existing entry if we already have a column
            named `name`.
        """
        self.validate_column(name, term)

        columns = self.columns
        if name in columns:
            if overwrite:
                self.remove(name)
            else:
                raise KeyError("Column '{}' already exists.".format(name))

        if not isinstance(term, ComputableTerm):
            raise TypeError(
                "{term} is not a valid pipeline column. Did you mean to "
                "append '.latest'?".format(term=term)
            )

        self._columns[name] = term

    @expect_types(name=str)
    def remove(self, name):
        """Remove a column.

        Parameters
        ----------
        name : str
            The name of the column to remove.

        Raises
        ------
        KeyError
            If `name` is not in self.columns.

        Returns
        -------
        removed : zipline.pipeline.Term
            The removed term.
        """
        return self.columns.pop(name)

    @expect_types(screen=Filter, overwrite=(bool, int))
    def set_screen(self, screen, overwrite=False):
        """Set a screen on this Pipeline.

        Parameters
        ----------
        filter : zipline.pipeline.Filter
            The filter to apply as a screen.
        overwrite : bool
            Whether to overwrite any existing screen.  If overwrite is False
            and self.screen is not None, we raise an error.
        """
        if self._screen is not None and not overwrite:
            raise ValueError(
                "set_screen() called with overwrite=False and screen already "
                "set.\n"
                "If you want to apply multiple filters as a screen use "
                "set_screen(filter1 & filter2 & ...).\n"
                "If you want to replace the previous screen with a new one, "
                "use set_screen(new_filter, overwrite=True)."
            )
        self._screen = screen

    def to_execution_plan(self, domain, default_screen, start_date, end_date):
        """
        Compile into an ExecutionPlan.

        Parameters
        ----------
        domain : zipline.pipeline.domain.Domain
            Domain on which the pipeline will be executed.
        default_screen : zipline.pipeline.Term
            Term to use as a screen if self.screen is None.
        all_dates : pd.DatetimeIndex
            A calendar of dates to use to calculate starts and ends for each
            term.
        start_date : pd.Timestamp
            The first date of requested output.
        end_date : pd.Timestamp
            The last date of requested output.

        Returns
        -------
        graph : zipline.pipeline.graph.ExecutionPlan
            Graph encoding term dependencies, including metadata about extra
            row requirements.
        """
        if self._domain is not GENERIC and self._domain is not domain:
            raise AssertionError(
                "Attempted to compile Pipeline with domain {} to execution "
                "plan with different domain {}.".format(self._domain, domain)
            )

        return ExecutionPlan(
            domain=domain,
            terms=self._prepare_graph_terms(default_screen),
            start_date=start_date,
            end_date=end_date,
        )

    def to_simple_graph(self, default_screen):
        """
        Compile into a simple TermGraph with no extra row metadata.

        Parameters
        ----------
        default_screen : zipline.pipeline.Term
            Term to use as a screen if self.screen is None.

        Returns
        -------
        graph : zipline.pipeline.graph.TermGraph
            Graph encoding term dependencies.
        """
        return TermGraph(self._prepare_graph_terms(default_screen))

    def _prepare_graph_terms(self, default_screen):
        """Helper for to_graph and to_execution_plan."""
        columns = self.columns.copy()
        screen = self.screen
        if screen is None:
            screen = default_screen
        columns[SCREEN_NAME] = screen
        return columns

    @expect_element(format=("svg", "png", "jpeg"))
    def show_graph(self, format="svg"):
        """
        Render this Pipeline as a DAG.

        Parameters
        ----------
        format : {'svg', 'png', 'jpeg'}
            Image format to render with.  Default is 'svg'.
        """
        g = self.to_simple_graph(AssetExists())
        if format == "svg":
            return g.svg
        elif format == "png":
            return g.png
        elif format == "jpeg":
            return g.jpeg
        else:
            # We should never get here because of the expect_element decorator
            # above.
            raise AssertionError("Unknown graph format %r." % format)

    @staticmethod
    @expect_types(term=Term, column_name=str)
    def validate_column(column_name, term):
        if term.ndim == 1:
            raise UnsupportedPipelineOutput(column_name=column_name, term=term)

    @property
    def _output_terms(self):
        """
        A list of terms that are outputs of this pipeline.

        Includes all terms registered as data outputs of the pipeline, plus the
        screen, if present.
        """
        terms = list(self._columns.values())
        screen = self.screen
        if screen is not None:
            terms.append(screen)
        return terms

    @expect_types(default=Domain)
    def domain(self, default):
        """
        Get the domain for this pipeline.

        - If an explicit domain was provided at construction time, use it.
        - Otherwise, infer a domain from the registered columns.
        - If no domain can be inferred, return ``default``.

        Parameters
        ----------
        default : zipline.pipeline.domain.Domain
            Domain to use if no domain can be inferred from this pipeline by
            itself.

        Returns
        -------
        domain : zipline.pipeline.domain.Domain
            The domain for the pipeline.

        Raises
        ------
        AmbiguousDomain
        ValueError
            If the terms in ``self`` conflict with self._domain.
        """
        # Always compute our inferred domain to ensure that it's compatible
        # with our explicit domain.
        inferred = infer_domain(self._output_terms)

        if inferred is GENERIC and self._domain is GENERIC:
            # Both generic. Fall back to default.
            return default
        elif inferred is GENERIC and self._domain is not GENERIC:
            # Use the non-generic domain.
            return self._domain
        elif inferred is not GENERIC and self._domain is GENERIC:
            # Use the non-generic domain.
            return inferred
        else:
            # Both non-generic. They have to match.
            if inferred is not self._domain:
                raise ValueError(
                    "Conflicting domains in Pipeline. Inferred {}, but {} was "
                    "passed at construction.".format(inferred, self._domain)
                )
            return inferred
