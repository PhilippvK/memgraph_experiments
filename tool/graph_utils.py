from pathlib import Path

import networkx as nx
from networkx.drawing.nx_agraph import write_dot


def graph_to_file(graph, dest, fmt="auto"):
    if not isinstance(dest, Path):
        dest = Path(dest)
    if fmt == "auto":
        fmt = dest.suffix[1:].upper()
    prog = "dot"
    if fmt == "DOT":
        write_dot(graph, dest)
    elif fmt in ["PDF", "PNG"]:
        graph = nx.nx_agraph.to_agraph(graph)
        graph.draw(dest, prog=prog)
    else:
        raise ValueError(f"Unsupported fmt: {fmt}")
