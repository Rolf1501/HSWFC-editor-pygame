from enum import Enum
from dataclasses import dataclass, field
from boundingbox import BoundingBox as BB
from coord import Coord

class Properties(Enum):
    ORTH = 0
    SYM = 1
    PAR = 2
    CENTERED_X = 3
    CENTERED_Y = 4
    CENTERED_Z = 5

class Cardinals(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

class Adjacency(Enum):
    NS = 0
    SN = 1
    EW = 2
    WE = 3

class Part:
    def __init__(self, size: BB, rotation: int) -> None:
        self.size = size
        self.rotation = rotation

MHNodeID = int

@dataclass()
class MHNode:
    shape_bb: BB = field() # index pointing to its original shape.
    node_id: MHNodeID = field(default=0)
    children: set[int] = field(default_factory={}) # direct children
    parent: MHNodeID = field(default=-1) # direct parents
    rotation: Coord = field(default_factory=Coord(0,0,0)) # in degrees, multiple of 90
    translation: Coord = field(default_factory=Coord(0,0,0))


@dataclass()
class MHEdge:
    frm: MHNodeID = field()
    to: MHNodeID = field()


@dataclass()
class MHLink:
    frm: MHNodeID = field()
    to: MHNodeID = field()
    adjacency: (Cardinals, Cardinals) = field() # Specifies which face of this should meet which face of that.
    # rotation: Coord = field(default_factory=Coord(0,0,0))
    properties: list[Properties] = field(default_factory=[])


@dataclass()
class MHTree:
    nodes: dict[MHNodeID, MHNode] = field(default_factory={})
    edges: list[MHEdge] = field(default_factory=[])
    links: list[MHLink] = field(default_factory=[])

    def add_node(self, node: MHNode):
        self.nodes[node.node_id] = node

    def add_link(self, link: MHLink):
        self.links.append(link)

    def add_edge(self, edge: MHEdge):
        self.edges.append(edge)

    # def propagate_rotation(self, from_node_id, to_node_id, rotation: Coord):
    #     assert (from_node_id in self.nodes and to_node_id in self.nodes)
    #     assert (self.nodes[from_node_id].parent == self.nodes[to_node_id].parent and self.nodes[from_node_id].parent >= 0)
        
    #     prop_rotation = self.nodes[from_node_id].rotation + rotation
    #     self.nodes[to_node_id].rotation += prop_rotation

    #     for child in self.nodes[to_node_id].children:
    #         self.propagate_rotation(to_node_id, child, prop_rotation)

    def initialize(self):
        # Establish parent child relationships in nodes.
        for edge in self.edges:
            assert (edge.frm in self.nodes and edge.to in self.nodes)
                # raise Exception(f"Nodes {edge.frm} and {edge.to} should both exist.")
            
            from_node = self.nodes[edge.frm]
            from_node.children.add(edge.to)
            
            to_node = self.nodes[edge.to]
            to_node.parent = edge.frm

        # Propagate rotations. Ensures that each node only has to be rotated once in the end.
        # for link in self.links:
        #     self.propagate_rotation(link.frm, link.to, link.rotation)






bbs = [
    BB(0,200,0,100),
    BB(0,150,0,50),
    BB(0,50,0,100)
]

nodes = {
    0: MHNode(bbs[0], 0),
    1: MHNode(bbs[1], 1),
    2: MHNode(bbs[2], 2)
}

edges = []
tree = MHTree()
