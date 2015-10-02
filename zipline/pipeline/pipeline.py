from zipline.utils.preprocess import expect_types, optional

from .term import Term
from .filters import Filter
from .graph import TermGraph


class Pipeline(object):
    """
    A computational Pipeline for use in trading algorithms.

    Parameters
    ----------
    columns : dict, optional
        Initial columns.
    screen : zipline.pipeline.term.Filter, optional
        Initial screen.

    Methods
    -------
    add
    remove
    set_screen

    Attributes
    ----------
    columns
    screen
    """
    __slots__ = ('_columns', '_screen', '__weakref__')

    @expect_types(
        columns=optional(dict),
        screen=optional(Filter),
    )
    def __init__(self, columns=None, screen=None):

        if columns is None:
            columns = {}
        self._columns = columns
        self._screen = screen

    @property
    def columns(self):
        """
        The columns currently applied to this pipeline.
        """
        return self._columns

    @property
    def screen(self):
        """
        The screen applied to the rows of this pipeline.
        """
        return self._screen

    @expect_types(term=Term, name=str)
    def add(self, term, name, overwrite=False):
        """
        Add a column.

        The results of computing `term` will show up as a column in the
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
        columns = self.columns
        if name in columns:
            if overwrite:
                self.remove(name)
            else:
                raise KeyError("Column '{}' already exists.".format(name))

        self._columns[name] = term

    @expect_types(name=str)
    def remove(self, name):
        """
        Remove a column.

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
        removed : zipline.pipeline.term.Term
            The removed term.
        """
        return self.columns.pop(name)

    @expect_types(screen=Filter)
    def set_screen(self, screen, overwrite=False):
        """
        Set a screen on this Pipeline.

        Parameter
        ---------
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

    def to_graph(self, screen_name, default_screen):
        """
        Compile into a TermGraph.

        Parameters
        ----------
        screen_name : str
            Name to supply for self.screen.
        default_screen : zipline.pipeline.term.Term
            Term to use as a screen if self.screen is None.
        """
        columns = self.columns.copy()
        screen = self.screen
        if screen is None:
            screen = default_screen
        columns[screen_name] = screen

        return TermGraph(columns)
