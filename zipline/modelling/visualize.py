"""
Tools for visualizing dependencies between Terms.
"""
from functools import partial
from contextlib import contextmanager
from logbook import Logger, StderrHandler
from networkx import topological_sort
from six import iteritems
import subprocess

from zipline.data.dataset import BoundColumn
from zipline.modelling import Filter, Factor, Classifier, Term
from zipline.modelling.engine import build_dependency_graph


logger = Logger('Visualize')


def delimit(delimiters, content):
    """
    Surround `content` with the first and last characters of `delimiters`.

    >>> delimit('[]', "foo")
    [foo]
    >>> delimit('""', "foo")
    '"foo"'
    """
    if len(delimiters) != 2:
        raise ValueError(
            "`delimiters` must be of length 2. Got %r" % delimiters
        )
    return ''.join([delimiters[0], content, delimiters[1]])


quote = partial(delimit, '""')
bracket = partial(delimit, '[]')


def begin_graph(f, name, **attrs):
    writeln(f, "strict digraph %s {" % name)
    writeln(f, "graph {attrs}".format(attrs=format_attrs(attrs)))


def begin_cluster(f, name, **attrs):
    attrs.setdefault("label", quote(name))
    writeln(f, "subgraph cluster_%s {" % name)
    writeln(f, "graph {attrs}".format(attrs=format_attrs(attrs)))


def end_graph(f):
    writeln(f, '}')


@contextmanager
def graph(f, name, **attrs):
    begin_graph(f, name, **attrs)
    yield
    end_graph(f)


@contextmanager
def cluster(f, name, **attrs):
    begin_cluster(f, name, **attrs)
    yield
    end_graph(f)


def roots(g):
    "Get nodes from graph G with indegree 0"
    return set(n for n, d in iteritems(g.in_degree()) if d == 0)


def leaves(g):
    "Get nodes from graph G with outdegree 0"
    return set(n for n, d in iteritems(g.out_degree()) if d == 0)


def write_term_graph(terms, filename, formats=('svg',)):
    """
    Write the dependency graph of `terms` as a dot graph.

    If `png` (default True), write a .png file using the system `dot` program.
    If `pdf` (default False), write a .pdf file using the system `dot` program.
    """
    g = build_dependency_graph(terms)
    dotfile = filename + '.dot'

    graph_attrs = {'rankdir': 'BT', 'splines': 'ortho'}
    cluster_attrs = {'style': 'filled', 'color': 'lightgoldenrod1'}

    with open(dotfile, 'w') as f:
        with graph(f, "G", **graph_attrs):

            # Write outputs cluster.
            with cluster(f, 'Outputs', **cluster_attrs):
                outputs = leaves(g)
                for term in outputs:
                    add_term_node(f, term)

            # Write inputs cluster.
            with cluster(f, 'Inputs', **cluster_attrs):
                inputs = roots(g)
                for term in inputs:
                    add_term_node(f, term)

            # Write intermediate results.
            for term in topological_sort(g):
                if term in inputs or term in outputs:
                    continue
                add_term_node(f, term)

            # Write edges
            for source, dest in g.edges():
                add_edge(f, id(source), id(dest))

    for format_ in formats:
        out = '.'.join([filename, format_])
        logger.info('Writing "%s"' % out)
        subprocess.call(['dot', '-T', format_, dotfile, '-o', out])


def writeln(f, s):
    f.write(s + '\n')


def fmt(obj):
    if isinstance(obj, Term):
        return '"{term}"'.format(term=obj.short_repr())
    return '"%s"' % obj


def add_term_node(f, term):
    declare_node(f, id(term), attrs_for_node(term))


def declare_node(f, name, attributes):
    writeln(f, "{0} {1};".format(name, format_attrs(attributes)))


def add_edge(f, source, dest):
    writeln(f, "{0} -> {1};".format(source, dest))


def attrs_for_node(term, **overrides):
    attrs = {
        'shape': 'box',
        'colorscheme': 'pastel19',
        'style': 'filled',
        'label': fmt(term),
    }
    if isinstance(term, BoundColumn):
        attrs['fillcolor'] = '1'
    if isinstance(term, Factor):
        attrs['fillcolor'] = '2'
    elif isinstance(term, Filter):
        attrs['fillcolor'] = '3'
    elif isinstance(term, Classifier):
        attrs['fillcolor'] = '4'

    attrs.update(**overrides or {})
    return attrs


def format_attrs(attrs):
    """
    Format key, value pairs from attrs into graphviz attrs format

    Example
    -------
    >>> format_attrs({'key1': 'value1', 'key2': 'value2'})
    '[key1=value1, key2=value2]'
    """
    if not attrs:
        return ''
    entries = ['='.join((key, value)) for key, value in iteritems(attrs)]
    return '[' + ', '.join(entries) + ']'


if __name__ == '__main__':
    from zipline.modelling.factor.technical import VWAP, MaxDrawdown, RSI
    from zipline.data.equities import USEquityPricing

    with StderrHandler():
        vwap = VWAP(window_length=5)
        dd_rank = MaxDrawdown([USEquityPricing.close], window_length=5).rank()
        rsi_rank = (-RSI()).rank()

        score = ((dd_rank + rsi_rank) / 2)

        write_term_graph(
            [
                vwap,
                score.percentile_between(0, 20),
            ],
            'temp',
        )
