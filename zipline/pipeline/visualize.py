"""
Tools for visualizing dependencies between Terms.
"""
from __future__ import unicode_literals

from contextlib import contextmanager
import errno
from functools import partial
from io import BytesIO
from subprocess import Popen, PIPE

from networkx import topological_sort
from six import iteritems

from zipline.pipeline.data import BoundColumn
from zipline.pipeline import Filter, Factor, Classifier, Term
from zipline.pipeline.term import AssetExists


class NoIPython(Exception):
    pass


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
    writeln(f, "graph {}".format(format_attrs(attrs)))


def begin_cluster(f, name, **attrs):
    attrs.setdefault("label", quote(name))
    writeln(f, "subgraph cluster_%s {" % name)
    writeln(f, "graph {}".format(format_attrs(attrs)))


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


def filter_nodes(include_asset_exists, nodes):
    if include_asset_exists:
        return nodes
    return filter(lambda n: n is not AssetExists(), nodes)


def _render(g, out, format_, include_asset_exists=False):
    """
    Draw `g` as a graph to `out`, in format `format`.

    Parameters
    ----------
    g : zipline.pipeline.graph.TermGraph
        Graph to render.
    out : file-like object
    format_ : str {'png', 'svg'}
        Output format.
    include_asset_exists : bool
        Whether to filter out `AssetExists()` nodes.
    """
    graph_attrs = {'rankdir': 'TB', 'splines': 'ortho'}
    cluster_attrs = {'style': 'filled', 'color': 'lightgoldenrod1'}

    in_nodes = g.loadable_terms
    out_nodes = list(g.outputs.values())

    f = BytesIO()
    with graph(f, "G", **graph_attrs):

        # Write outputs cluster.
        with cluster(f, 'Output', labelloc='b', **cluster_attrs):
            for term in filter_nodes(include_asset_exists, out_nodes):
                add_term_node(f, term)

        # Write inputs cluster.
        with cluster(f, 'Input', **cluster_attrs):
            for term in filter_nodes(include_asset_exists, in_nodes):
                add_term_node(f, term)

        # Write intermediate results.
        for term in filter_nodes(include_asset_exists, topological_sort(g)):
            if term in in_nodes or term in out_nodes:
                continue
            add_term_node(f, term)

        # Write edges
        for source, dest in g.edges():
            if source is AssetExists() and not include_asset_exists:
                continue
            add_edge(f, id(source), id(dest))

    cmd = ['dot', '-T', format_]
    try:
        proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    except OSError as e:
        if e.errno == errno.ENOENT:
            raise RuntimeError(
                "Couldn't find `dot` graph layout program. "
                "Make sure Graphviz is installed and `dot` is on your path."
            )
        else:
            raise

    f.seek(0)
    proc_stdout, proc_stderr = proc.communicate(f.read())
    if proc_stderr:
        raise RuntimeError(
            "Error(s) while rendering graph: %s" % proc_stderr.decode('utf-8')
        )

    out.write(proc_stdout)


def display_graph(g, format='svg', include_asset_exists=False):
    """
    Display a TermGraph interactively from within IPython.
    """
    try:
        import IPython.display as display
    except ImportError:
        raise NoIPython("IPython is not installed.  Can't display graph.")

    if format == 'svg':
        display_cls = display.SVG
    elif format in ("jpeg", "png"):
        display_cls = partial(display.Image, format=format, embed=True)

    out = BytesIO()
    _render(g, out, format, include_asset_exists=include_asset_exists)
    return display_cls(data=out.getvalue())


def writeln(f, s):
    f.write((s + '\n').encode('utf-8'))


def fmt(obj):
    if isinstance(obj, Term):
        if hasattr(obj, 'short_repr'):
            r = obj.short_repr()
        else:
            r = type(obj).__name__
    else:
        r = obj
    return '"%s"' % r


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
