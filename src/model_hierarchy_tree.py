from dataclasses import dataclass, field
from boundingbox import BoundingBox as BB
from coord import Coord
from util_data import Cardinals, Dimensions, Operations
import networkx as nx

@dataclass
class Properties:
    operation: Operations = field()
    dimension: Dimensions = field(default=Dimensions.X)

@dataclass
class Adjacency:
    frm: Cardinals = field()
    to: Cardinals = field()

MHNodeID = int

@dataclass()
class MHNode:
    shape_bb: BB = field() # index pointing to its original shape.
    node_id: MHNodeID = field(default=0)
    children: set[int] = field(default_factory={}) # direct children
    parent: MHNodeID = field(default=-1) # direct parents
    rotation: Coord = field(default_factory=Coord(0,0,0)) # in degrees, multiple of 90
    translation: Coord = field(default_factory=Coord(0,0,0))
    origin: Coord = field(init=False, default_factory=Coord(0,0,0))


@dataclass()
class MHEdge:
    frm: MHNodeID = field()
    to: MHNodeID = field()


@dataclass()
class MHLink:
    source: MHNodeID = field()
    attachment: MHNodeID = field()
    adjacency: Cardinals # Specifies which face of this should meet which face of that.
    # rotation: Coord = field(default_factory=Coord(0,0,0))
    properties: list[Properties] = field(default_factory=[])
    relative_adjacency: bool = field(default=True)

@dataclass
class MHTree(nx.Graph):
    nodes: dict[MHNodeID, MHNode] = field(default_factory={})
    edges: list[MHEdge] = field(default_factory=[])
    links: list[MHLink] = field(default_factory=[])
    # def __init__(self):
    #     pass
    def add_node(self, node: MHNode):
        self.nodes[node.node_id] = node

    def add_link(self, link: MHLink):
        self.links.append(link)

    def add_edge(self, edge: MHEdge):
        self.edges.append(edge)

    def initialize(self):
        # Establish parent child relationships in nodes.
        for edge in self.edges:
            assert (edge.frm in self.nodes and edge.to in self.nodes)
                # raise Exception(f"Nodes {edge.frm} and {edge.to} should both exist.")
            
            from_node = self.nodes[edge.frm]
            from_node.children.add(edge.to)
            
            to_node = self.nodes[edge.to]
            to_node.parent = edge.frm