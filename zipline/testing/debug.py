import sys
import subprocess
import networkx as nx


def debug_mro_failure(name, bases, output_file):
    graph = build_linearization_graph(name, bases)
    cycles = sorted(nx.cycles.simple_cycles(graph), key=len)
    cycle = cycles[0]

    if output_file is not None:
        sys.__stderr__.write("Writing debug graph to {}\n".format(output_file))
        nx.write_dot(graph.subgraph(cycle), output_file)
        subprocess.call(['dot', '-T', 'svg', '-O', output_file])

    # Return a nicely formatted error describing the cycle.
    lines = ["Cycle found when trying to compute MRO for {}:\n".format(name)]
    for source, dest in list(zip(cycle, cycle[1:])) + [(cycle[-1], cycle[0])]:
        label = verbosify_label(graph.get_edge_data(source, dest)['label'])
        lines.append("{} comes before {}: cause={}"
                     .format(source, dest, label))

    return '\n'.join(lines)


def build_linearization_graph(child_name, bases):
    g = nx.DiGraph()
    _build_linearization_graph(g, type(child_name, (object,), {}), bases)
    return g


def _build_linearization_graph(g, child, bases):
    add_implicit_edges(g, child, bases)
    add_direct_edges(g, child, bases)


def add_direct_edges(g, child, bases):
    # Enforce that bases are ordered in the order that the appear in child's
    # class declaration.
    g.add_path([b.__name__ for b in bases], label=child.__name__ + '(O)')

    # Add direct edges.
    for base in bases:
        g.add_edge(child.__name__, base.__name__, label=child.__name__ + '(D)')
        add_direct_edges(g, base, base.__bases__)


def add_implicit_edges(g, child, bases):
    # Enforce that bases' previous linearizations are preserved.
    for base in bases:
        g.add_path(
            [b.__name__ for b in base.mro()],
            label=base.__name__ + '(L)',
        )


VERBOSE_LABELS = {
    "(D)": "(Direct Subclass)",
    "(O)": "(Parent Class Order)",
    "(L)": "(Linearization Order)",
}


def verbosify_label(label):
    prefix = label[:-3]
    suffix = label[-3:]
    return " ".join([prefix, VERBOSE_LABELS[suffix]])
