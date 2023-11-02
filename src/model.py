from boundingbox import BoundingBox as BB
from model_hierarchy_tree import Properties, MHLink, Operations
from util_data import Cardinals, Dimensions, Operations
from dataclasses import dataclass, field
from model_tree import ModelTree
import math
from coord import Coord
from boundingbox import BoundingBox as BB
import networkx as nx

class Part:
    def __init__(self, size: BB, rotation: int=0, translation: Coord=Coord(0,0)) -> None:
        self.size = size
        self.rotation = rotation
        self.translation = translation
    
    def absolute_north(self) -> Cardinals:
        # Assumes rotation expressed in degrees
        return Cardinals((self.rotation / 90.0) % len(Cardinals))
    

class Model:  
    def __init__(self, parts: dict[int, Part], links: list[MHLink], model_tree: ModelTree=None):
        self.parts = parts
        self.links = links
        if model_tree is None:
            self.model_tree = ModelTree(self.parts.keys(), [(l.source, l.attachment) for l in self.links if l.adjacency is not None])
        else:
            self.model_tree = model_tree
        self.model_tree_cuts_memoi: dict[(int,int), list[int]] = {}
        self._precalc_tree_cut_sets()

    def _precalc_tree_cut_sets(self):
        for l in self.links:
            if (l.source, l.attachment) in self.model_tree.edges:
                self.model_tree_cuts_memoi[(l.source, l.attachment)] = self.model_tree.get_attachment_subtree(l.source, l.attachment)
            else:
                # If the two nodes are not adjacent, the attachment group can be found by performing a cut between the attachment and penultimate node in the shortest path from source to attachment.
                shortest_path = nx.shortest_path(self.model_tree, l.source, l.attachment)
                subtree = self.model_tree.get_attachment_subtree(shortest_path[-2], shortest_path[-1])
                self.model_tree_cuts_memoi[(l.source, l.attachment)] = subtree


    def solve(self):
        self._determine_final_rotations()
        self._rotate_all()

        self._determine_adjacency_translations()
        self._determine_property_translations()
        self._translate_all()

    def _translate_all(self):
        for k in self.parts.keys():
            self.parts[k].size = self._translate_part(self.parts[k])
    
    @staticmethod
    def _translate_part(part: Part) -> BB:
        t_x = part.translation.x
        t_y = part.translation.y
        return part.size + BB(t_x, t_x, t_y, t_y)

    def _determine_adjacency_translations(self):
        for l in self.links:
            if l.adjacency is not None:
                t = self._adjacency_translation(l.source, l.attachment, l.adjacency)
                for n in self.model_tree_cuts_memoi[(l.source, l.attachment)]:
                    self.parts[n].translation += t
    
    def _determine_property_translations(self):
        for l in self.links:
            if l.adjacency is not None:
                for p in l.properties:
                    if p.operation == Operations.CENTER:
                        t = self._center_translation(l.source, l.attachment, p.dimension)
                        for n in self.model_tree_cuts_memoi[(l.source, l.attachment)]:
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
            subtree_nodes = self.model_tree_cuts_memoi[(l.source, l.attachment)]
            source = self.parts[l.source]
            attachment = self.parts[l.attachment]

            rotation_diff = abs(source.rotation - attachment.rotation)
            for p in l.properties:
                rotation = 0
                if p.operation == Operations.ORTH:
                    if rotation_diff != 90:
                        rotation = 90 # rotation in degrees

                elif p.operation == Operations.SYM:
                    if rotation_diff != 180:
                        rotation = 180

                if rotation != 0:
                    # Rotation needs to be propagated to all parts attached to the attachment.
                    for n in subtree_nodes:
                        self.parts[n].rotation += rotation
    
    def _rotate_all(self):
        # self._determine_final_rotations()
        for k in self.parts.keys():
            self.parts[k].size = self._rotate_part(self.parts[k])
        

    @staticmethod
    def _rotate_part(part: Part) -> BB:
        """
        Currently done in 2D. Needs extension for 3D later
        """
        bb = part.size
        rotation = part.rotation

        # Same as translation to the origin
        size_vector = Coord(bb.width(), bb.height())

        # rotation: [[cos a, -sin a], [sin a, cos a]]  [x, y]
        # rotated vector: (x * cos a - y * sin a), (x * sin a + y * cos a)

        rotation_norm = rotation % 360
        rad = math.radians(rotation_norm)
        cosa = math.cos(rad)
        sina = math.sin(rad)
        rot_coord = Coord(size_vector.x * cosa - size_vector.y * sina, size_vector.x * sina + size_vector.y * cosa)
        
        trans_rot_coord = rot_coord + Coord(bb.minx, bb.miny)

        rot_bb = BB(
            int(math.ceil(min(bb.minx, trans_rot_coord.x))),
            int(math.ceil(max(0, trans_rot_coord.x))),
            int(math.ceil(min(bb.miny, trans_rot_coord.y))),
            int(math.ceil(max(0, trans_rot_coord.y))))
        
        rot_bb += rot_bb.to_positive_translation()

        return rot_bb
                

    def _handle_adjacency(self, source: Part, attachment: Part, adjacency: Cardinals, relative_adjacency=True):
        """
        Aligns the attachments on the face of source indicated by the adjacency.
        """

        # The links are relative, so need to take the source's rotation into account.
        if relative_adjacency:
            adjacency = Cardinals((adjacency.value + source.absolute_north().value) % len(Cardinals))

        if adjacency == Cardinals.EAST:
            attachment.size.maxx = source.size.maxx + attachment.size.width()
            attachment.size.minx = source.size.maxx

        elif adjacency == Cardinals.WEST:
            attachment.size.minx = source.size.minx - attachment.size.width()
            attachment.size.maxx = source.size.minx

        elif adjacency == Cardinals.NORTH:
            attachment.size.miny = source.size.miny - attachment.size.height()
            attachment.size.maxy = source.size.miny

        elif adjacency == Cardinals.SOUTH:
            attachment.size.maxy = source.size.maxy + attachment.size.height()
            attachment.size.miny = source.size.maxy
        
        return source, attachment

    def _center_translation(self, source_id: int, attachment_id: int, dimension: Dimensions) -> Coord:
        """
        Centers attachment relative to source.
        """
        source = self.parts[source_id]
        attachment = self.parts[attachment_id]

        source_center = source.size.center()
        attachment_center = attachment.size.center()
        diff_center = source_center - attachment_center
        # In case the part has been rotated such that the relative axes do not align with the global x and y axes, switch the dimension.
        if source.absolute_north() == Cardinals.EAST or source.absolute_north() == Cardinals.WEST:
            dimension = Dimensions((dimension.value + 1) % len(Dimensions))
        if dimension == Dimensions.X:
            # bb2_width = bb2.width()
            # bb2.minx = bb1_center.x - 0.5 * bb2_width
            # bb2.maxx = bb2.minx + bb2_width
            return Coord(diff_center.x, 0)
        elif dimension == Dimensions.Y:
            return Coord(0, diff_center.y)
            # bb2_height = bb2.height()
            # bb2.miny = bb1_center.y - 0.5 * bb2_height
            # bb2.maxy = bb2.miny + bb2_height

        return Coord(0,0,0)

    def _handle_properties(self, source: Part, attachment: Part, properties: list[Properties]):
        source_n = source
        attachment_n = attachment

        for prop in properties:
            if prop.operation == Operations.CENTER:
                source_n.size, attachment_n.size = self._center(source, attachment, prop.dimension)
                
        return source_n, attachment_n