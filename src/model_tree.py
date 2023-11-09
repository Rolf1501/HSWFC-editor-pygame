import networkx as nx
from collections import namedtuple
from part import Part
from model_hierarchy_tree import MHLink

Edge = namedtuple("Edge", ["u", "v"])

class ModelTree(nx.DiGraph):
    def __init__(self, nodes: list[int]=[], links: list[MHLink]=[], incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.add_nodes_from(nodes)
        self.add_edges_from([(l.source, l.attachment) for l in links if l.adjacency is not None])
        self.links = links


    @classmethod
    def from_parts(cls, parts: dict[int, Part], links: list[MHLink]):
        return ModelTree(parts.keys(), links)
   

    def get_attachment_subtree(self, source: int, attachment: int) -> list[int]:
        # Only one edge needs to be removed to create two disjoint sets, as a tree is acyclic.
        temp_model = self.to_undirected()
        if temp_model.has_edge(source, attachment):
            temp_model.remove_edge(source, attachment)
        return nx.dfs_tree(temp_model, attachment).nodes


    def get_sibling_order(self):
        # The sibling order is determined by which nodes other nodes depend on.
        # Each link (u, v) is a dependency between nodes u and v, where v depends on u.
        # The root is the that has no incoming links.
        g = self
        
        nodes = sorted(list(g.nodes), key=lambda n: g.in_degree[n])
                
        if len(nodes) == 0:
            return []
        
        root = nodes[0]

        order = []
        stack = []
        stack.append(root)

        # The sibling order can be found with DFS.
        while len(stack) > 0:
            current_node = stack.pop()
            order.append(current_node)
            for succ in g.successors(current_node):
                if succ not in order: # ensure that each node is visited at most once.
                    stack.append(succ)

        return order
        
