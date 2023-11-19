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


# @dataclass()
# class MHNode:
#     name: str
#     node_id: MHNodeID
# children: set[int] = field(default_factory={}) # direct children
# parent: MHNodeID = field(default=-1) # direct parent


@dataclass()
class MHEdge:
    frm: MHNodeID
    to: MHNodeID


@dataclass()
class MHLink:
    source: MHNodeID = field()
    attachment: MHNodeID = field()
    adjacency: Cardinals  # Specifies which face of this should meet which face of that.
    # rotation: Coord = field(default_factory=Coord(0,0,0))
    properties: list[Properties] = field(default_factory=[])
    relative_adjacency: bool = field(default=True)


class MHTree(nx.DiGraph):
    # nodes: dict[MHNodeID, MHNode] = field(default_factory={})
    # edges: list[MHEdge] = field(default_factory=[])
    # links: list[MHLink] = field(default_factory=[])
    def __init__(self, mh_nodes: list[MHNode], mh_edges: list[MHEdge], incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.mh_nodes = mh_nodes
        self.add_nodes_from([node.node_id for node in mh_nodes])
        self.add_edges_from([(edge.frm, edge.to) for edge in mh_edges])

    # def get_parent(self, node: int):
    #     return self.pred[node] if self.has_node(node) else None
    # def add_node(self, node: MHNode):
    #     self.nodes[node.node_id] = node

    # def add_nodes(self, nodes: list[MHNode]):
    #     for node in nodes:
    #         self.add_node(node)

    # def add_link(self, link: MHLink):
    #     self.links.append(link)

    # def add_links(self, links: list[MHLink]):
    #     for link in links:
    #         self.add_link(link)

    # def add_edge(self, edge: MHEdge):
    #     self.edges.append(edge)

    # def add_edges(self, edges: list[MHEdge]):
    #     for edge in edges:
    #         self.add_edge(edge)

    # def initialize(self):
    #     # Establish parent child relationships in nodes.
    #     for edge in self.edges:
    #         assert (edge.frm in self.nodes and edge.to in self.nodes)
    #             # raise Exception(f"Nodes {edge.frm} and {edge.to} should both exist.")

    #         from_node = self.nodes[edge.frm]
    #         from_node.children.add(edge.to)

    #         to_node = self.nodes[edge.to]
    #         to_node.parent = edge.frm
