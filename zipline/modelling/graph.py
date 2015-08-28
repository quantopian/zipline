"""
FFC-specific extensions to networkx.DiGraph
"""
from networkx import (
    DiGraph,
    topological_sort,
)
from six import itervalues, iteritems


class CyclicDependency(Exception):
    pass


class TermGraph(DiGraph):
    """
    Graph represention of FFC Term dependencies

    Each node in the graph has an `extra_rows` attribute, indicating how many,
    if any, extra rows we should compute for the node.  Extra rows are most
    often needed when a term is an input to a rolling window computation.  For
    example, if we compute a 30 day moving average of price from day X to day
    Y, we need to load price data for the range from day (X - 29) to day Y.

    Parameters
    ----------
    terms : dict
        A dict mapping names to terms.

    Attributes
    ----------
    outputs
    extra_rows
    max_extra_rows

    Methods
    -------
    ordered()
        Return a topologically-sorted iterator over the terms in self.
    """
    def __init__(self, terms):
        super(TermGraph, self).__init__(self)
        parents = set()
        for term in itervalues(terms):
            self._add_to_graph(term, parents, extra_rows=0)
            # No parents should be left between top-level terms.
            assert not parents

        self._outputs = terms
        self._extra_rows = {
            term: attrs['extra_rows']
            for term, attrs in iteritems(self.node)
        }
        self._max_extra_rows = max(itervalues(self._extra_rows))
        self._ordered = topological_sort(self)

    @property
    def outputs(self):
        """
        Dict mapping names to designated output terms.
        """
        return self._outputs

    @property
    def extra_rows(self):
        """
        Dict mapping term -> number of extra rows to compute for term.
        """
        return self._extra_rows

    @property
    def max_extra_rows(self):
        """
        Maximum number of extra rows required to compute any term in the graph.
        """
        return self._max_extra_rows

    def ordered(self):
        """
        Return a topologically-sorted iterator over the terms in `self`.
        """
        return iter(self._ordered)

    def _add_to_graph(self, term, parents, extra_rows):
        """
        Add `term` and all its inputs to the graph.
        """
        # If we've seen this node already as a parent of the current traversal,
        # it means we have an unsatisifiable dependency.  This should only be
        # possible if the term's inputs are mutated after construction.
        if term in parents:
            raise CyclicDependency(term)
        parents.add(term)

        try:
            existing = self.node[term]
        except KeyError:
            # `term` is not yet in the graph: add it with the specified number
            # of extra rows.
            self.add_node(term, extra_rows=extra_rows)
        else:
            # `term` is already in the graph because we've been traversed by
            # another parent.  Ensure that we have enough extra rows to satisfy
            # all of our parents.
            existing['extra_rows'] = max(extra_rows, existing['extra_rows'])

        extra_rows_for_subterms = extra_rows + term.extra_input_rows
        for subterm in term.inputs:
            self._add_to_graph(
                subterm,
                parents,
                extra_rows=extra_rows_for_subterms
            )
            self.add_edge(subterm, term)

        parents.remove(term)
