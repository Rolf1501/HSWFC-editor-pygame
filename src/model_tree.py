import networkx as nx
from collections import namedtuple
Edge = namedtuple("Edge", ["u", "v"])

class ModelTree(nx.Graph):
    def __init__(self, nodes: list[int]=[], edges: list[(Edge)]=[], incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.add_nodes_from(nodes)
        self.add_edges_from(edges)
        # assert nx.is_tree(self)
    
    def get_attachment_subtree(self, source: int, attachment: int) -> list[int]:
        # Only one edge needs to be removed to create two disjoint sets, as a tree is acyclic.
        temp_model = self.copy()
        temp_model.remove_edge(source, attachment)
        return nx.dfs_tree(temp_model, attachment).nodes
        
