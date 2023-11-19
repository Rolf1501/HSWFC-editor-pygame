from communicator import Communicator as C, Verbosity as V
from part import Part
import model_hierarchy_tree
from model_tree import ModelTree
from model import Model

comm = C()


class GeometricSolver:
    def __init__(self, mht: model_hierarchy_tree.MHTree, parts: dict[int, Part], full_model_tree: ModelTree):
        self.mht = mht
        self.parts = parts
        self.full_model_tree = full_model_tree

    def process(self, node_id: int, processed: dict[int, bool]):

        part = self.parts[node_id]
        part_info = (node_id, part.name)
        comm.communicate(f"Currently collapsing {part_info}...")
        children = list(self.mht.successors(node_id))

        # If the current node has children.
        if children:

            # Determine collapse order of children.
            comm.communicate(f"Determining sibling processing order...")

            sibling_links = [link for link in self.full_model_tree.links if
                             link.source in children and link.attachment in children]
            model_subtree = ModelTree(incoming_graph_data=self.full_model_tree.subgraph(children), links=sibling_links)

            # Processing order corresponds to the dependencies of the siblings.
            process_order = model_subtree.get_sibling_order()
            if process_order:
                comm.communicate(f"Found sibling order: {list(map(lambda sib: (sib, self.parts[sib].name), process_order))}",
                                 V.HIGH)
            else:
                comm.communicate(f"No more siblings found for {part_info}", V.HIGH)

            # Arrange all children.
            # Make sure children inherit translation and rotation from parent.
            for k in children:
                self.parts[k].rotation += self.parts[node_id].rotation
                self.parts[k].translation += self.parts[node_id].translation

            # Only include parts and links of the children.
            model = Model(
                {k: self.parts[k] for k in children},
                sibling_links,
                model_subtree)

            model.solve()
            model.contain_in(part.extent)

            # Process each child following the sibling order.
            for node in process_order:
                self.process(node, processed)

            # TODO: after all children have been processed, assemble here.
            # This is also the place where the tight fits can be made.
            # Make use of properties stored in the link between the two parts.
            # E.g. when aligning with center: move attachment to source along centerline until overlap occurs.
            processed[node_id] = True

            comm.communicate(f"All children of node {part_info} completed. Fitting parts...")
            comm.communicate(f"Parts status:", V.HIGH)
            for p in self.parts.items():
                comm.communicate(f"\t{p}", V.HIGH)

        else:
            comm.communicate("No more children found. Checking adjacent siblings processing status...")
            # TODO: collapse meta node into leaves.
            # If no other adjacent siblings have been processed yet, simply collapse.
            # Otherwise, check where that sibling is, how it related to this node and where it overlaps.
            # That determines the initial seeds/tiles for this part.
            # Collapse this section and return.
            # Return to parent when all children are processed.
            processed[node_id] = True

        comm.communicate(f"Processing node {(node_id, self.parts[node_id].name)} complete.")
        return True
