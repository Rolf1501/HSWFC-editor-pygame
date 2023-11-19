from dataclasses import dataclass, field
from offset import Offset
from typing import Any, Set, Dict, Tuple
from input import Input
import numpy as np

@dataclass(frozen=True)
class Adjacency:
    """
    Data class used for storing an adjacency, mostly exists for convenience so it can be pretty printed via the __repr__ override below. 
    
    Used by MetaNode.
    """
    from_tile: int
    to_tile: int
    offset: Offset

    def __repr__(self):
        return f"{self.from_tile}-{self.to_tile}"

@dataclass(frozen=True)
class  MetaNode:
    """
    The basic element used for structuring the meta-tree.
    
    This class has multiple convenience methods built in to quickly get certain elements from the DAG relative to this node.
    
    NOTE: the structure employed here differs a little bit from what is in paper: 
          where in the paper every tile relates to exactly one node, here we can 
          have multiple tiles relating to the same node. The most bottom
          MetaNodes contain the tile IDs of the leaf/concrete tiles, so there are 
          no explicit leaf nodes here. This means that the entire meta-tree consists
          of nodes corresponding to a meta-tile.
          
          It is possible to specify multiple meta-tiles in the input for a single meta-node, 
          but this is untried, and does not seem very useful :)
    """
    name: str  # name of the node
    archetypes: Dict[Tuple[Any, Any], Any] = field(compare=False, init=False, default_factory=dict)  # i.e. the parents
    subtypes: Dict[Tuple[Any, Any], Any] = field(compare=False, init=False, default_factory=dict)  # i.e. the children
    tile_ids: Set[int] = field(compare=False, default_factory=set)  # the indices of the tiles that this node represents
    inputs: Set[Input] = field(compare=False, default_factory=set)  # the input images that correspond to this node
    adjacencies: Dict[Offset, Set[Adjacency]] = field(compare=False, default_factory=dict)  # The adjacencies that were found in the input images of this node
    properties: Dict[str, Any] = field(compare=False, default_factory=dict)  # Other properties, parsed from a json file (see input format)

    
    def add_link(self, link):
        """
        Add a MetaLink. Used by the MetaTreeBuilder.
        """
        if link.frm is self:
            self.subtypes[link.key()] = link
        elif link.to is self:
            self.archetypes[link.key()] = link
    
    
    def remove_link(self, link):
        """
        Remove a MetaLink. Used by the MetaTreeBuilder.
        """
        if link.frm is self:
            del self.subtypes[link.key()]
        elif link.to is self:
            del self.archetypes[link.key()]
               
    @property
    def leaves(self):
        """
        Returns the leaves of the sub-meta-tree that has this node at its root/source.
        """
        S = set()
        if not self.subtypes:
            S.add(self)
        for subtype_link in self.subtypes.values():
            S.update(subtype_link.to.leaves)
        return S
    
    @property
    def nodes(self):
        """
        Returns the sub-meta-tree with this node at its root/source, including this node.
        """
        S = {self}
        for subtype_link in self.subtypes.values():
            S.update(subtype_link.to.nodes)
        return S
    
    @property
    def ancestry(self):
        """
        Returns all the ancestors that can lead to this node.
        """
        S = {self}
        for archetype_link in self.archetypes.values():
            S.update(archetype_link.frm.ancestry)
        return S
    
#     def root(self):
#         node = self
#         while node.parent:
#             node = node.parent
#         return node    
    
    def __repr__(self):
        return f"{self.name}:{self.tile_ids}"



@dataclass(frozen=True)
class MetaLink:
    """
    This class is used for specifying a link/edge between two nodes
    """
    frm: MetaNode  # The "from" node
    to: MetaNode  # The "to" node
    properties: Dict[str, Any] = field(compare=False, default_factory=dict)  # The same properties json that is in MetaNode
    
    # NOTE: The reason why the edges also keep a copy, is because the JSON also contains the probability weights.
    #       The DAG is built from a tree-ish folder structure, where folders/nodes get merged in case of equivalent names.
    #       Still though, the probability of reaching such a node can differ per edge/link, hence we need to store this 
    #       information per edge/link.
    
    
    def key(self):
        """
        MetaLinks are uniquely identified by the from/to node pair, because we never want multiple edges between the same (ordered) pair of nodes.
        
        One could argue that this should be a set, since we never want both possible pairs either (circular dependency), but this requires more
        advanced checking anyway for the non-trivial cases, which should happen elsewhere.
        """
        return self.frm, self.to
    
    def __repr__(self):
        return f"{self.frm.name}-->{self.to.name}"

@dataclass(frozen=True)
class MetaTreeBuilder:
    """
    This class is responsible for putting a tree together from MetaNodes and MetaLinks. It also holds some global information,
    such as: a list of all the nodes, a list of all the tiles, a list of all the links, and all the metamasks.
    """
    nodes: Dict[MetaNode, MetaNode] = field(compare=False, init=False, default_factory=dict)  # Dict of all nodes (NOTE: Practically only the name field of a MetaNode is used for the dictionary, so string-indexing would've worked too here, #lazy)
    links: Dict[Tuple[MetaNode, MetaNode], MetaLink] = field(compare=False, init=False, default_factory=dict)  # Dict of all edges/links, indexed by MetaNode pairs for easy retrieval
    tiles: Dict[int, MetaNode] = field(compare=False, init=False, default_factory=dict)  # Dict of all tiles, indexed by tile index
    metamasks: Dict[int, Any] = field(compare=False, init=False, default_factory=dict)  # Like the child/subtree masks, this is a boolean indexing mask that highlights which tiles are in the subtree of a meta-node, INCLUDING itself
    
    # TODO: Might be better to just implement a root method on the nodes, though this is way cheaper
    @property
    def root(self):
        """
        Gets the root of the meta-tree. This does require the root folder to be called "root"...
        
        Note that MetaNodes are uniquely identified by their name (check the properties of the fields in MetaNode, all others have 'compare' on False)
        """
        return self.nodes[MetaNode("root")]
        
    @property
    def tile_ids(self):
        """
        Gets the (unordered) set of all tile indices.
        """
        return set(self.tiles.keys())

    
    def add_node(self, node):
        """
        Adds a MetaNode to the meta-tree.
        """
        self.nodes[node] = node
    
    
    def add_link(self, frm, to, properties):
        """
        Creates a link between the two given MetaNodes with the given properties, and ensures linkage in the tree and between the nodes is setup properly, and rebuilds the tree.
        """
        link = MetaLink(frm, to, properties)
        self.nodes[frm] = frm
        self.nodes[to] = to
        self.links[link.key()] = link
        frm.add_link(link)
        to.add_link(link)
        self.rebuild_tile_dict()
    
    
    def remove_link(self, frm, to):
        """
        Removes the link (there should be only one) between the two given MetaNodes, if it exists, and rebuilds the tree.
        """
        link_key = (frm, to)
        link = self.links[link_key]
        del self.links[link_key]
        frm.remove_link(link)
        to.remove_link(link)
        self.rebuild_tile_dict()


    def rebuild_tile_dict(self):
        """
        Rebuilds the tile dictionary based on the current dict of MetaNodes.
        """
        self.tiles.clear()
        for node in self.nodes:
            for tile in node.tile_ids:
                self.tiles[tile] = node

    def build_metamasks(self):
        """
        Builds all the metamasks.
        """
        self.metamasks.clear()
        for root in self.nodes:
            for tile in root.tile_ids:
                self.metamasks[tile] = np.full(len(self.tiles), False)
                for node in root.nodes - {root}:
                    for subtile in node.tile_ids:
                        self.metamasks[tile][subtile] = True      
                
    def __repr__(self):
        return f"nodes: {list(self.nodes.values())}\nlinks: {list(self.links.values())}"
    
   