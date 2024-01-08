from dataclasses import dataclass, field
from util_data import Cardinals, Dimensions, Operations
import networkx as nx
from collections import namedtuple


@dataclass
class Properties:
    operation: Operations = field()
    dimension: Dimensions = field(default=Dimensions.X)


@dataclass
class Adjacency:
    frm: Cardinals = field()
    to: Cardinals = field()


MHNodeID = int
MHNode = namedtuple("MHNode", ["node_id", "name"])


@dataclass()
class MHEdge:
    frm: MHNodeID
    to: MHNodeID


@dataclass()
class MHLink:
    source: MHNodeID = field()
    attachment: MHNodeID = field()
    adjacency: Cardinals  # Specifies which face of this should meet which face of that.
    properties: list[Properties] = field(default_factory=[])
    relative_adjacency: bool = field(default=True)


class MHTree(nx.DiGraph):
    def __init__(
        self,
        mh_nodes: list[MHNode],
        mh_edges: list[MHEdge],
        incoming_graph_data=None,
        **attr
    ):
        super().__init__(incoming_graph_data, **attr)
        self.mh_nodes = mh_nodes
        self.add_nodes_from([node.node_id for node in mh_nodes])
        self.add_edges_from([(edge.frm, edge.to) for edge in mh_edges])
