import pydot
from nn_tools.types import GraphType


def graph2dot(network_graph: GraphType) -> pydot.Graph:
    edges = []
    for layer, (_, parents) in network_graph.items():
        for pa in parents:
            edges.append((pa, layer))
    return pydot.graph_from_edges(edges, directed=True)
