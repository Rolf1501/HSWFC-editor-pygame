from dataclasses import dataclass, field
import numpy as np
from offsets import OffsetFactory, Offset
from bidict import bidict
from collections import namedtuple
from properties import Properties
from terminal import Terminal
from coord import Coord


class Relation(namedtuple("Relation", ["other", "weight"])):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()


class Adjacency:
    def __init__(
        self,
        source: int,
        allowed_neighbours: set[Relation],
        offset: Offset,
        symmetric: bool,
        properties: list[Properties] = [],
    ) -> None:
        self.source = source
        self.allowed_neighbours = allowed_neighbours
        self.offset = offset
        self.symmetric = symmetric
        self.properties = properties


class AdjacencyAny(Adjacency):
    def __init__(
        self, source: int, offset: Offset, symmetric: bool, weight: float
    ) -> None:
        super().__init__(source, {}, offset, symmetric)
        self.weight = weight


@dataclass
class AdjacencyMatrix:
    parts: list[int]
    part_adjacencies: set[Adjacency]
    terminals: list[Terminal] = field(default=None)
    offsets_dimensions: int = field(default=3)
    adjacencies: set[Adjacency] = field(init=False)
    offsets: list[Offset] = field(init=False)
    ADJ: dict[Offset, np.ndarray] = field(init=False)
    ADJ_W: dict[Offset, np.ndarray] = field(init=False)
    parts_to_index_mapping: bidict[int, int] = field(init=False)

    atom_mapping: bidict[int, tuple] = field(init=False)
    atom_adjacency_matrix: dict[Offset, np.ndarray] = field(init=False)
    atom_adjacency_matrix_w: dict[Offset, np.ndarray] = field(init=False)
    part_atom_range_mapping: dict = field(init=False)

    def __post_init__(self):
        self.atom_mapping = bidict()
        self.atom_adjacency_matrix = {}
        self.atom_adjacency_matrix_w = {}

        self.part_atom_range_mapping = {}

        n_parts = len(self.parts)
        self.parts_to_index_mapping = bidict({self.parts[i]: i for i in range(n_parts)})

        self.offsets = OffsetFactory().get_offsets(
            dimensions=self.offsets_dimensions, cardinal=True
        )
        self.atom_adjacencies()

        self.ADJ = {}
        self.ADJ_W = {}
        for offset in self.offsets:
            self.ADJ[offset] = np.full((n_parts, n_parts), False)
            self.ADJ_W[offset] = np.full((n_parts, n_parts), 0.0)

    def get_n_atoms(self):
        return len(self.atom_mapping.keys())

    def atom_adjacencies(self):
        n_atoms = 0

        for p in self.parts:
            t = self.terminals[p]
            self.part_atom_range_mapping[p] = (n_atoms, n_atoms + t.n_atoms)
            n_atoms += t.n_atoms

        for o in self.offsets:
            self.atom_adjacency_matrix[o] = np.full((n_atoms, n_atoms), False)
            self.atom_adjacency_matrix_w[o] = np.full((n_atoms, n_atoms), 0.0)

        self.inner_atom_adjacency()

        self.atom_atom_adjacency()

        # TODO: base calculation on heightmaps.
        # for p in self.part_adjacencies:
        #     s = p.source
        #     o = p.offset
        #     axis = None
        #     if o.x != 0:
        #         axis = 1
        #     elif o.y != 0:
        #         axis = 0
        #     else:
        #         axis = 2
        #     print(f"AXIS: {axis}")

        # s_heightmaps = self.terminals[s].heightmaps
        # s_offset_hm = s_heightmaps[o]
        # observer_window = np.full(s_offset_hm.shape, )
        # offsets = Offset()
        # for n in p.allowed_neighbours:
        #     t = n.other
        #     t_heightmaps = self.terminals[t].heightmaps
        #     # Get the complementing offset
        #     t_offset_hm = t_heightmaps[o.scaled(-1)]
        # for k in self.atom_adjacency_matrix.keys():
        #     print(k, self.atom_adjacency_matrix[k])
        # print(self.atom_mapping)

    def inner_atom_adjacency(self):
        """
        Formalizes the atom atom adjacencies within a molecule.
        """
        for p in self.parts:
            t = self.terminals[p]

            for i in t.atom_indices:
                k = len(self.atom_mapping.keys())
                self.atom_mapping[k] = (p, i)
            # Make sure that the atoms of the same part have to be together.
            # An adjacency constraint, only allowing the other neighbouring atom of the same part atom enforces this.
            for i in t.atom_indices:
                for o in self.offsets:
                    this_index = self.atom_mapping.inverse[(p, i)]

                    # Not all atoms have a neigbour of the same part in all offsets.
                    # The try-except block catches invalid combinations.
                    try:
                        other_index = self.atom_mapping.inverse[(p, i + o)]
                    except:
                        continue
                    self.atom_adjacency_matrix[o][this_index, other_index] = True
                    # Mirror the operation
                    self.atom_adjacency_matrix[o.scaled(-1)][
                        other_index, this_index
                    ] = True

                    self.atom_adjacency_matrix_w[o][this_index, other_index] = 1.0
                    # Mirror the operation
                    self.atom_adjacency_matrix_w[o.scaled(-1)][
                        other_index, this_index
                    ] = 1.0

    def atom_atom_heightmap_adjacency(self):
        for a in self.part_adjacencies:
            this = self.terminals[a.source]
            that = [self.terminals[n.other] for n in a.allowed_neighbours]
            for t in that:
                pass

        pass

    def atom_adjacency_from_heightmaps(
        self,
        this_heightmap: np.ndarray,
        that_heightmap: np.ndarray,
        this_max_depth: int,
        that_max_depth: int,
    ):
        """
        Finds the atom adjacencies based on the heightmaps.
        Slides that heightmap over this heightmap and find the minimal offset such that the surfaces represented by the heightmaps touch each other.
        """
        (this_y, this_x) = this_heightmap.shape
        (that_y, that_x) = that_heightmap.shape

        # By selecting the smaller heightmap as the base, the number of checks for touching surfaces is reduced.
        if this_x * this_y < that_x * that_y:
            base = this_heightmap
            slider = that_heightmap
            base_depth = this_max_depth
            slider_depth = that_max_depth
            max_x_base = this_x
            max_y_base = this_y
            max_x_slider = that_x
            max_y_slider = that_y
        else:
            base = that_heightmap
            slider = this_heightmap
            base_depth = that_max_depth
            slider_depth = this_max_depth
            max_x_base = that_x
            max_y_base = that_y
            max_x_slider = this_x
            max_y_slider = this_y

        n_shifts_x = this_x + that_x - 1
        n_shifts_y = this_y + that_y - 1

        # The flipped slider allows for aligning the two heightmaps more easily. Prevents having to deal with index manipulation.
        flipped = np.flip(slider)

        for y in range(n_shifts_y):
            # The slider slides over the base. The range covered in the base depends on the slider.
            y_base_s, y_base_e = self.sliding_range(y, max_y_base, max_y_slider)

            # The slider slides over the base. The range covered in the slider depends on the base.
            y_slider_s, y_slider_e = self.sliding_range(y, max_y_slider, max_y_base)

            for x in range(n_shifts_x):
                # The slider slides over the base. The range covered in the base depends on the slider.
                x_base_s, x_base_e = self.sliding_range(x, max_x_base, max_x_slider)

                # The slider slides over the base. The range covered in the slider depends on the base.
                x_slider_s, x_slider_e = self.sliding_range(x, max_x_slider, max_x_base)

                base_window = base[y_base_s:y_base_e, x_base_s, x_base_e]
                slider_window = flipped[y_slider_s:y_slider_e, x_slider_s:x_slider_e]

                # TODO: infer range of empty space from heightmaps in other directions.

                # If either window is empty, indicated by max distance to occupied cell, then there is no surface for contact.
                if np.all(base_window >= base_depth) or np.all(
                    slider_window >= slider_depth
                ):
                    continue

                # Otherwise, take the minimal distance between potential contact points.
                distance = base_window + slider_window
                min_distance_coord = np.argmin(distance, axis=1)

                # TODO: translate the masks onto each other depending on the found distance.

        return

    def sliding_range(self, index, max_this, max_that):
        """
        Returns the range such that it is within this is covered by that, given the index for offset.
        """
        # The start cannot be negative.
        start = max(0, index - max_that + 1)

        # The end cannot exceed the bounds of the this window.
        end = min(max_this, index)
        return start, end

    def atom_atom_adjacency(self):
        """
        Determines the adjacency of atoms of different parts.
        For cuboids, each atom of part A at the meeting faces may be adjacent to any atom of part B at the opposite face and vice versa.
        """
        for a in self.part_adjacencies:
            source = a.source
            offset = a.offset
            allowed_neighbours = a.allowed_neighbours
            source_wdh = self.terminals[source].extent.whd()
            n_dims = len(source_wdh)

            # Infer the axis from the offset.
            offset_index = np.abs(np.array(offset)).argmax()
            offset_complement = offset.scaled(-1)

            # The direction is signed. The meeting face is the last slice in a positive direction. The first slice otherwise.
            source_slice = (
                source_wdh[offset_index] - 1 if offset[offset_index] == 1 else 0
            )

            for n in allowed_neighbours:
                other: int = n.other
                other_wdh = self.terminals[other].extent.whd()
                other_slice = (
                    0 if offset[offset_index] == 1 else other_wdh[offset_index] - 1
                )

                # Find the pair of axes whose adjacency is determined. Those are the axes that are not in the inferred direction.
                for i in range(n_dims):
                    if i == offset_index:
                        continue

                    for j in range(i + 1, n_dims):
                        if j == offset_index:
                            continue

                        for s_0 in range(source_wdh[i]):
                            for s_1 in range(source_wdh[j]):
                                for o_0 in range(other_wdh[i]):
                                    for o_1 in range(other_wdh[j]):
                                        s_c = self.get_relative_atom_coord(
                                            n_dims,
                                            (i, s_0),
                                            (j, s_1),
                                            (offset_index, source_slice),
                                        )
                                        s_index = self.get_atom_index(source, s_c)

                                        o_c = self.get_relative_atom_coord(
                                            n_dims,
                                            (i, o_0),
                                            (j, o_1),
                                            (offset_index, other_slice),
                                        )
                                        o_index = self.get_atom_index(
                                            other, Coord(*o_c)
                                        )

                                        self.atom_adjacency_matrix[offset][
                                            s_index, o_index
                                        ] = True
                                        self.atom_adjacency_matrix[offset_complement][
                                            o_index, s_index
                                        ] = True

                                        self.atom_adjacency_matrix_w[offset][
                                            s_index, o_index
                                        ] = n.weight
                                        self.atom_adjacency_matrix_w[offset_complement][
                                            o_index, s_index
                                        ] = n.weight
                    i = j  # Stop the looping when a pair has been found.

    def get_atom_index(self, part_id, coord):
        return self.atom_mapping.inverse[(part_id, coord)]

    def get_relative_atom_coord(self, n_dims, *tuples):
        """
        Returns the coord of an atom within the part it belongs to.
        """
        assert len(tuples) == n_dims
        c = np.full(n_dims, None)
        for t in tuples:
            c[t[0]] = t[1]

        return Coord(*c)

    def allow_adjacencies(self, adjs: set[Adjacency]):
        for adj in adjs:
            neg_offset = adj.offset.negation()
            source_i = self.parts_to_index_mapping[adj.source]
            if isinstance(adj, AdjacencyAny):
                self.ADJ[adj.offset][source_i] = self.get_full(True)
                self.ADJ_W[adj.offset][source_i] = self.get_full(adj.weight)
                # Set the columns.
                if adj.symmetric:
                    self.ADJ[neg_offset][:, source_i] = True
                    self.ADJ_W[neg_offset][:, source_i] = adj.weight
            else:
                for neighbour in adj.allowed_neighbours:
                    neighbour_i = self.parts_to_index_mapping[neighbour.other]
                    self.ADJ[adj.offset][source_i, neighbour_i] = True
                    self.ADJ_W[adj.offset][source_i, neighbour_i] = neighbour.weight
                    if adj.symmetric:
                        self.ADJ[neg_offset][neighbour_i, source_i] = True
                        self.ADJ_W[neg_offset][neighbour_i, source_i] = neighbour.weight

    def get_adj(self, offset: Offset, choice_id: int):
        return self.atom_adjacency_matrix[offset][choice_id]

    def get_adj_w(self, offset: Offset, choice_id: int):
        return self.atom_adjacency_matrix_w[offset][choice_id]

    def print_adj(self):
        for o, a in self.ADJ.items():
            print(o)
            for i in a:
                print(i)

    def get_full(self, value):
        return np.full((self.get_n_atoms()), value)
