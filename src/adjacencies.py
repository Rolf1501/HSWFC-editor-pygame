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

        self.atom_atom_heightmap_adjacency()
        self.print_adj()

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
            for allowed_neighbour in a.allowed_neighbours:
                self.min_distance_diffs_from_heightmaps(
                    a.source,
                    allowed_neighbour.other,
                    a.offset,
                    allowed_neighbour.weight,
                )
        pass

    def min_distance_diffs_from_heightmaps(
        self, this_id: int, that_id: int, offset: Offset, weight: float
    ):
        """
        Finds the atom adjacencies based on the heightmaps.
        Slides that heightmap over this heightmap and find the minimal offset such that the surfaces represented by the heightmaps touch each other.
        """

        def calc_slider_range(max_d_slider, max_d_base, d):
            """
            Calculates the slider sliding window range, for a given shift d.
            """
            # Do not exceed the slider window's bounds.
            # When the slider's coverage is smaller than the base window y, the base window should not exceed the y length of the slider.

            # There are two main cases: (1) d <= max_d_slider and (2) d > max_d_slider.
            # (1) indicates the case where the overlap of the two windows is a subset of the base.
            # (2) indicates the moment when the base index >= sliding starting index.
            # In case (1), max_d_slider - d is positive and corresponds to the first index where the sliding window moves over the base window.
            # In case (2), the former would be negative, which outside the bounds. Hence, cap it off at 0.
            start_d = max(0, max_d_slider - d)

            # Same cases as before. In case (1), the last index of the slider still overlaps with an index of the base. Hence, the cap.
            # In case (2), the length of the slider window should not exceed the length of the base.
            # +1 is to compensate for numpy range handling.
            end_d = min(max_d_slider, max_d_slider + max_d_base - d) + 1
            return start_d, end_d

        def calc_base_range(max_d_base, start_d_slider, end_d_slider, d):
            """
            Calculates the base sliding window range, for a given shift d.
            Slider window range is used to infer base window range, since the two should not differ in length.
            """
            # Stays 0 until y is larger than the slider, at which point the base start should start to increase as well.
            start_d_base = max(d - end_d_slider, 0)

            # Do not exceed the slider window's bounds.
            # When the slider's coverage is smaller than the base window y, the base window should not exceed the y length of the slider.
            # Note that the second argument does not contain +1, since this is already incorporated in the end slider range.
            end_d_base = min(
                max_d_base + 1, start_d_base + end_d_slider - start_d_slider
            )
            return start_d_base, end_d_base

        this_terminal = self.terminals[this_id]
        that_terminal = self.terminals[that_id]
        this_heightmap = this_terminal.heightmaps[offset]
        that_heightmap = that_terminal.heightmaps[offset.complement()]
        offset_direction_index = np.nonzero([offset.y, offset.x, offset.z])[0][0]
        this_max_depth = this_terminal.mask.shape[offset_direction_index]
        that_max_depth = that_terminal.mask.shape[offset_direction_index]
        (this_y, this_x) = this_heightmap.shape
        (that_y, that_x) = that_heightmap.shape

        # By selecting the smaller heightmap as the base, the number of checks for touching surfaces is reduced.
        if this_x * this_y <= that_x * that_y:
            base_terminal = this_terminal
            slider_terminal = that_terminal
            base = this_heightmap
            slider = that_heightmap
            base_depth = this_max_depth
            slider_depth = that_max_depth
            max_x_base = this_x
            max_y_base = this_y
            max_x_slider = that_x
            max_y_slider = that_y
            base_id = this_id
            slider_id = that_id
            print(offset)
            print(f"base_id: {base_id}; slider_id: {slider_id}")
            print(f"base hm: {base}")
            print(f"slider hm: {slider}")
            print(base_terminal.mask)
            print(slider_terminal.mask)
        else:
            base_terminal = that_terminal
            slider_terminal = this_terminal
            # Switch the offset, since the base and slider are swapped compared to the input.
            offset = offset.complement()
            base = that_heightmap
            slider = this_heightmap
            base_depth = that_max_depth
            slider_depth = this_max_depth
            max_x_base = that_x
            max_y_base = that_y
            max_x_slider = this_x
            max_y_slider = this_y
            base_id = that_id
            slider_id = this_id

        base_dims = base_terminal.mask.shape
        slider_dims = slider_terminal.mask.shape

        n_shifts_x = this_x + that_x - 1
        n_shifts_y = this_y + that_y - 1

        max_x_index_slider = max_x_slider - 1
        max_x_index_base = max_x_base - 1
        max_y_index_slider = max_y_slider - 1
        max_y_index_base = max_y_base - 1

        for y in range(n_shifts_y):
            # start_y_slider = max(0, max_y_index_slider - y)  # Offset by y.

            # # Do not exceed the slider window's bounds.
            # # The window may not exceed the y length of the base window.
            # end_y_slider = (
            #     min(max_y_index_slider, max_y_index_base + max_y_index_slider - y)
            #     + 1
            #     # min(max_y_index_slider, start_y_slider + max_y_index_base) + 1
            # )
            start_y_slider, end_y_slider = calc_slider_range(
                max_y_index_slider, max_y_index_base, y
            )

            # Stays 0 until y is larger than the slider, at which point the base start should start to increase as well.
            # start_y_base = max(y - end_y_slider, 0)

            # end_y_base = (
            #     min(max_y_index_base, start_y_base + end_y_slider - start_y_slider)
            #     + 1
            #     # min(max_y_base - 1, start_y_base + end_y_slider - start_y_slider) + 1
            # )
            start_y_base, end_y_base = calc_base_range(
                max_y_index_base, start_y_slider, end_y_slider, y
            )
            for x in range(n_shifts_x):
                # start_x_slider = max(0, max_x_index_slider - x)  # Offset by x.

                # # Do not exceed the slider window's bounds.
                # # The window may not exceed the x length of the base window.
                # end_x_slider = (
                #     min(max_x_index_slider, start_x_slider + max_x_index_base) + 1
                # )
                start_x_slider, end_x_slider = calc_slider_range(
                    max_x_index_slider, max_x_index_slider, x
                )

                start_x_base, end_x_base = calc_base_range(
                    max_x_index_base,
                    start_x_slider,
                    end_x_slider,
                    x,
                )

                # # Stays 0 until x is larger than the slider, at which point the base start should start to increase as well.
                # start_x_base = max(x - end_x_slider + 1, 0)

                # # Do not exceed the slider window's bounds.
                # # When the slider's coverage is smaller than the base window x, the base window should not exceed the x length of the slider.
                # end_x_base = (
                #     min(max_x_base - 1, start_x_base + end_x_slider - start_x_slider)
                #     + 1
                # )

                base_window = base[start_y_base:end_y_base, start_x_base:end_x_base]
                slider_window = slider[
                    start_y_slider:end_y_slider, start_x_slider:end_x_slider
                ]

                # TODO: infer range of empty space from heightmaps in other directions.

                # If either window is empty, indicated by max distance to occupied cell, then there is no surface for contact.
                if np.all(base_window >= base_depth) or np.all(
                    slider_window >= slider_depth
                ):
                    print(f"\nCONTINUING: bw {base_window} sw {slider_window}\n")
                    continue

                # Otherwise, take the minimal distance between potential contact points by overlapping the windows.
                distance = base_window + slider_window
                min_distance = np.min(distance)

                # Relative to the base.
                sign = offset.to_numpy_array(True)[offset_direction_index]

                # In case of 0 distance, need to compare two slices.
                # For each additional distance, need one more slice from each terminal.

                base_slice_indices = np.array(
                    [(start_y_base, end_y_base), (start_x_base, end_x_base)]
                )

                slider_slice_indices = np.array(
                    [(start_y_slider, end_y_slider), (start_x_slider, end_x_slider)]
                )

                if sign < 0:
                    # Insert the depth specification in the direction corresponding to the offset.
                    base_slice_indices = np.insert(
                        base_slice_indices,
                        offset_direction_index,
                        (0, 1 + min_distance),
                        axis=0,
                    )

                    slider_slice_indices = np.insert(
                        slider_slice_indices,
                        offset_direction_index,
                        (
                            slider_dims[offset_direction_index] - min_distance - 1,
                            slider_dims[offset_direction_index],
                        ),
                        axis=0,
                    )
                else:
                    # Insert the depth specification in the direction corresponding to the offset.
                    base_slice_indices = np.insert(
                        base_slice_indices,
                        offset_direction_index,
                        (
                            base_dims[offset_direction_index] - min_distance - 1,
                            base_dims[offset_direction_index],
                        ),
                        axis=0,
                    )

                    slider_slice_indices = np.insert(
                        slider_slice_indices,
                        offset_direction_index,
                        (0, 1 + min_distance),
                        axis=0,
                    )

                # Slice values correspond to one another. i.e. base slice index 0,0 corresponds to slider slice index 0,0

                # The meeting slices are distance d apart. So, slice 0 meets slice -d: (-1 - sign) * (-d), and -1 meets d: sign * (-1 + d). Depends on sign!
                # 1 meets -1: (1 - d); -1 meets 1
                # this means that the meeting slices are aligned when one of them is flipped in the offset direction.
                base_atoms = base_terminal.atom_coord_mask
                slider_atoms = slider_terminal.atom_coord_mask
                b_s_y = base_slice_indices[0]
                b_s_x = base_slice_indices[1]
                b_s_z = base_slice_indices[2]
                s_s_y = slider_slice_indices[0]
                s_s_x = slider_slice_indices[1]
                s_s_z = slider_slice_indices[2]

                insert_value = (
                    slider_dims[offset_direction_index]
                    if sign < 0
                    else base_dims[offset_direction_index]
                )
                base_to_slider_coord = np.insert(
                    np.array(
                        [
                            start_y_base - start_y_slider,
                            start_x_base - start_x_slider,
                        ]
                    ),
                    offset_direction_index,
                    -sign * (min_distance - insert_value),
                )

                print(f"\nBSCOORD: {base_to_slider_coord}")
                print(sign, min_distance, slider_dims, offset_direction_index)
                print(f"Shifts: x: {x} y: {y}")
                print(f"bsx: {b_s_x} bsy: {b_s_y}: bsz: {b_s_z}")
                print(f"ssx: {s_s_x} ssy: {s_s_y} ssz: {s_s_z}")
                print(f"glob offset {offset}")
                print(f"base_dims: {base_dims}")
                print(f"slider_dims: {slider_dims}")
                print(f"bw: {base_window} sw: {slider_window}")

                for i_y in range(base_dims[0]):
                    for i_x in range(base_dims[1]):
                        for i_z in range(base_dims[2]):
                            # for i_y in range(b_s_y[1] - b_s_y[0]):
                            #     for i_x in range(b_s_x[1] - b_s_x[0]):
                            #         for i_z in range(b_s_z[1] - b_s_z[0]):
                            # Select a coord in base.
                            b_c_coord = i_y, i_x, i_z
                            # b_c_coord = i_y + b_s_y[0], i_x + b_s_x[0], i_z + b_s_z[0]
                            b_c = base_atoms[b_c_coord[0], b_c_coord[1], b_c_coord[2]]
                            # print(f"bccoord: {b_c_coord}")
                            # And check if an atom is present.
                            if b_c is not None:
                                # If so: loop over the possible neighbours in the slider.
                                for o in self.offsets:
                                    # Base starting coord relative to the slider's position.

                                    rel_offset = np.asarray(b_c_coord) + [
                                        o.y,
                                        o.x,
                                        o.z,
                                    ]
                                    s_coord = np.array(
                                        rel_offset - base_to_slider_coord, dtype=int
                                    )

                                    # Only consider coords within bounds.
                                    if np.any(s_coord < 0) or np.any(
                                        s_coord >= slider_dims
                                    ):
                                        # print(
                                        #     f"Coord out of bounds... {s_coord} {o} {b_c}"
                                        # )
                                        continue

                                    s_c = slider_atoms[
                                        s_coord[0], s_coord[1], s_coord[2]
                                    ]
                                    if s_c is not None:
                                        # Found adj!
                                        base_atom_index = self.get_atom_index(
                                            base_id, b_c
                                        )
                                        slider_atom_index = self.get_atom_index(
                                            slider_id, s_c
                                        )
                                        if self.update_atom_adjacency(
                                            base_atom_index,
                                            slider_atom_index,
                                            o,
                                            weight,
                                        ):
                                            print(
                                                f"""FOUND IT! b_c: {base_id}, {b_c} s_c: {slider_id}, {s_c} o: {o},
                                                glob_offset: {offset}; shift: {y,x};
                                                min_dist: {min_distance}; s_coord: {s_coord}
                                                rel_offset: {rel_offset}"""
                                            )
                            else:
                                print(f"Passing bc: {b_c}")

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
                                        self.update_atom_adjacency(
                                            s_index, o_index, offset, n.weight
                                        )

                    i = j  # Stop the looping when a pair has been found.

    def get_atom_index(self, part_id, coord):
        return self.atom_mapping.inverse[(part_id, coord)]

    def update_atom_adjacency(self, this_index, that_index, offset: Offset, weight):
        if (
            not self.atom_adjacency_matrix[offset][this_index, that_index]
            and not self.atom_adjacency_matrix[offset.complement()][
                that_index, this_index
            ]
        ):
            self.atom_adjacency_matrix[offset][this_index, that_index] = True
            self.atom_adjacency_matrix[offset.complement()][
                that_index, this_index
            ] = True

            self.atom_adjacency_matrix_w[offset][this_index, that_index] = weight
            self.atom_adjacency_matrix_w[offset.complement()][
                that_index, this_index
            ] = weight

            print(
                f"Added adjacency:\n\t{this_index} ({self.atom_mapping[this_index]}), {that_index} ({self.atom_mapping[that_index]}), {offset}, w: {weight}"
            )
            return True
        return False

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
        print(self.atom_mapping)
        print(self.part_atom_range_mapping)
        for o, a in self.atom_adjacency_matrix.items():
            print(o)
            for i in a:
                print(np.asarray(i, dtype=int))

    def get_full(self, value):
        return np.full((self.get_n_atoms()), value)
