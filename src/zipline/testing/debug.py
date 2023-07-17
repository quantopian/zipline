import os
import subprocess

import networkx as nx


def debug_mro_failure(name, bases):
    graph = build_linearization_graph(name, bases)
    cycles = sorted(nx.cycles.simple_cycles(graph), key=len)
    cycle = cycles[0]

    if os.environ.get("DRAW_MRO_FAILURES"):
        output_file = name + ".dot"
    else:
        output_file = None

    # Return a nicely formatted error describing the cycle.
    lines = ["Cycle found when trying to compute MRO for {}:\n".format(name)]
    for source, dest in list(zip(cycle, cycle[1:])) + [(cycle[-1], cycle[0])]:
        label = verbosify_label(graph.get_edge_data(source, dest)["label"])
        lines.append("{} comes before {}: cause={}".format(source, dest, label))

    # Either graphviz graph and tell the user where it went, or tell people how
    # to enable that feature.
    lines.append("")
    if output_file is None:
        lines.append(
            "Set the DRAW_MRO_FAILURES environment variable to"
            " render a GraphViz graph of this cycle."
        )
    else:
        try:
            nx.write_dot(graph.subgraph(cycle), output_file)
            subprocess.check_call(["dot", "-T", "svg", "-O", output_file])
            lines.append("GraphViz rendering written to " + output_file + ".svg")
        except Exception as e:
            lines.append("Failed to write GraphViz graph. Error was {}".format(e))

    return "\n".join(lines)


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
    g.add_path([b.__name__ for b in bases], label=child.__name__ + "(O)")

    # Add direct edges.
    for base in bases:
        g.add_edge(child.__name__, base.__name__, label=child.__name__ + "(D)")
        add_direct_edges(g, base, base.__bases__)


def add_implicit_edges(g, child, bases):
    # Enforce that bases' previous linearizations are preserved.
    for base in bases:
        g.add_path(
            [b.__name__ for b in base.mro()],
            label=base.__name__ + "(L)",
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
