from boundingbox import BoundingBox as BB
from model_hierarchy_tree import MHLink, Operations
from util_data import Cardinals, Dimensions, Operations
from model_tree import ModelTree
import math
from coord import Coord
from boundingbox import BoundingBox as BB
import networkx as nx
from part import Part
from communicator import Communicator, Verbosity as V
from util_data import MIN, MAX
import numpy as np


comm = Communicator()

class Model:  
    def __init__(self, parts: dict[int, Part], links: list[MHLink], model_tree: ModelTree=None):
        self.parts = parts
        self.links = links
        if model_tree is not None:
            self.model_tree = model_tree
        else:
            self.model_tree = ModelTree.from_parts(parts, links)
        
        assert nx.is_forest(self.model_tree)
        
        self.model_subtrees_memoi: dict[(int,int), list[int]] = {}
        self._precalc_subtree_nodes_sets()


    def _precalc_subtree_nodes_sets(self):
        comm.communicate("Deriving subtrees...")
        for l in self.links:
            if (l.source, l.attachment) in self.model_tree.edges:
                self.model_subtrees_memoi[(l.source, l.attachment)] = self.model_tree.get_attachment_subtree(l.source, l.attachment)
            else:
                # If the two nodes are not adjacent, the attachment group can be found by performing 
                # a cut between the attachment and penultimate node in the shortest path from source to attachment.
                shortest_path = nx.shortest_path(self.model_tree, l.source, l.attachment)
                subtree = self.model_tree.get_attachment_subtree(shortest_path[-2], shortest_path[-1])
                self.model_subtrees_memoi[(l.source, l.attachment)] = subtree
        comm.communicate("Subtree derivations complete.")


    def solve(self):
        self._determine_final_rotations()
        self._rotate_all()

        self._determine_adjacency_translations()
        self._determine_property_translations()
        self._translate_all()

        return self.parts


    def contain_in(self, container: BB):
        """
        Tries to fit all parts inside the container.
        """
        bb: BB = BB(MAX, MIN, MAX, MIN, auto_adjust=False)
        for p in self.parts:
            part_bb = self.parts[p].size
            bb.minx = min(part_bb.minx, bb.minx)
            bb.maxx = max(part_bb.maxx, bb.maxx)
            bb.miny = min(part_bb.miny, bb.miny)
            bb.maxy = max(part_bb.maxy, bb.maxy)

        if container.can_contain(bb):
            translation = container.min_coord() - bb.min_coord()
        else:
            translation = container.center() - bb.center()
        for k in self.parts:
            self.parts[k].size.translate(translation)

    def _translate_all(self):
        for k in self.parts.keys():
            self.parts[k].size = self._translate_part(self.parts[k])


    @staticmethod
    def _translate_part(part: Part) -> BB:
        return part.size.translate(part.translation)


    def _determine_adjacency_translations(self):
        for l in self.links:
            if l.adjacency is not None:
                t = self._adjacency_translation(l.source, l.attachment, l.adjacency)
                for n in self.model_subtrees_memoi[(l.source, l.attachment)]:
                    self.parts[n].translation += t
    

    def _determine_property_translations(self):
        for l in self.links:
            if l.adjacency is not None:
                for p in l.properties:
                    if p.operation == Operations.CENTER:
                        t = self._center_translation(l.source, l.attachment, p.dimension)
                        for n in self.model_subtrees_memoi[(l.source, l.attachment)]:
                            self.parts[n].translation += t


    def _adjacency_translation(self, source_id: int, attachment_id: int, adjacency: Cardinals, relative_adjacency=True):
        # The links are relative, so need to take the source's rotation into account.
        translation = Coord(0,0)
        source = self.parts[source_id]
        attachment = self.parts[attachment_id]

        if relative_adjacency:
            adjacency = Cardinals((adjacency.value + source.absolute_north().value) % len(Cardinals))

        if adjacency == Cardinals.EAST:
            translation.x = source.size.maxx - attachment.size.minx

        elif adjacency == Cardinals.WEST:
            translation.x = source.size.minx - attachment.size.maxx

        elif adjacency == Cardinals.NORTH:
            translation.y = source.size.miny - attachment.size.maxy

        elif adjacency == Cardinals.SOUTH:
            translation.y = source.size.maxy - attachment.size.miny
        
        return translation


    def _determine_final_rotations(self):
        # Determine final rotations of all parts.
        rotation_diff = 0
        for l in self.links:
            subtree_nodes = self.model_subtrees_memoi[(l.source, l.attachment)]
            source = self.parts[l.source]
            attachment = self.parts[l.attachment]

            rotation_diff = abs(source.rotation - attachment.rotation)
            for p in l.properties:
                rotation = 0
                if p.operation == Operations.ORTH:
                    if rotation_diff != 90:
                        rotation = 90

                elif p.operation == Operations.SYM:
                    if rotation_diff != 180:
                        rotation = 180

                if rotation != 0:
                    # Rotation needs to be propagated to all parts attached to the attachment.
                    comm.communicate(f"Propagating rotation to attachments subtree {subtree_nodes} of part {l.attachment}", V.LOW)
                    for n in subtree_nodes:
                        self.parts[n].rotation += rotation


    def _rotate_all(self):
        for k in self.parts.keys():
            self.parts[k].size = self.rotate_part_bb(self.parts[k].size, self.parts[k].rotation)
        

    @staticmethod
    def rotate_part_bb(bb: BB, rotation: int, origin: Coord = Coord(0,0,0)) -> BB:
        """
        Currently done in 2D. Needs extension for 3D later
        """


        # rotation: [[cos a, -sin a], [sin a, cos a]]  [x, y]
        # rotated vector: (x * cos a - y * sin a), (x * sin a + y * cos a)

        # TODO: memoi/precalc the cos and sin outcomes.
        rotation_norm = rotation % 360
        rad = math.radians(rotation_norm)
        cosa = math.cos(rad)
        sina = math.sin(rad)

        rot_matrix = np.asarray([[cosa, -sina], [sina, cosa]])
        
        min_coord = (bb.min_coord() - origin).to_numpy_array() 
        max_coord = (bb.max_coord() - origin).to_numpy_array()
        min_rot_coord = np.matmul(rot_matrix, min_coord)
        max_rot_coord = np.matmul(rot_matrix, max_coord)

        rot_bb = BB(
            min(min_rot_coord[0], max_rot_coord[0]),
            max(min_rot_coord[0], max_rot_coord[0]),
            min(min_rot_coord[1], max_rot_coord[1]),
            max(min_rot_coord[1], max_rot_coord[1]),
        ).translate(origin)
        
        return rot_bb
                

    def _center_translation(self, source_id: int, attachment_id: int, dimension: Dimensions) -> Coord:
        """
        Centers attachment relative to source.
        """
        source = self.parts[source_id]
        attachment = self.parts[attachment_id]

        source_center = source.size.center()
        attachment_center = attachment.size.center()
        diff_center = source_center - attachment_center
        # In case the part has been rotated such that the local axes do not align with the global axes, switch the dimension.
        if source.absolute_north() == Cardinals.EAST or source.absolute_north() == Cardinals.WEST:
            dimension = Dimensions((dimension.value + 1) % len(Dimensions))
        if dimension == Dimensions.X:
            return Coord(diff_center.x, 0)
        elif dimension == Dimensions.Y:
            return Coord(0, diff_center.y)
        
        return Coord(0,0,0)