from dataclasses import dataclass, field
import numpy as np
from offsets import OffsetFactory, Offset
from bidict import bidict
from collections import namedtuple

class Relation(namedtuple("Relation", ["other","weight"])):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()
    

class Adjacency:
    def __init__(self, source: int, allowed_neighbours: set[Relation], offset: Offset, symmetric: bool) -> None:
        self.source = source
        self.allowed_neighbours = allowed_neighbours
        self.offset = offset
        self.symmetric = symmetric


@dataclass
class AdjacencyMatrix:
    parts: list[int]
    adjacencies: set[Adjacency]
    offsets_dimensions: int = field(default=3)
    offsets: np.matrix = field(init=False)
    ADJ: dict[Offset, np.ndarray] = field(init=False)
    ADJ_W: dict[Offset, np.ndarray] = field(init=False)
    parts_to_index_mapping: bidict[int, int] = field(init=False)

    def __post_init__(self):
        n_parts = len(self.parts)
        self.parts_to_index_mapping = bidict({self.parts[i]: i for i in range(n_parts)})
        self.offsets = OffsetFactory().get_offsets(dimensions=self.offsets_dimensions, cardinal=True)
        self.ADJ = {}
        self.ADJ_W = {}
        for offset in self.offsets:
            self.ADJ[offset] = np.full((n_parts, n_parts), False)
            self.ADJ_W[offset] = np.full((n_parts, n_parts), 0.0)
        self.allow_adjacencies(self.adjacencies)


    def allow_adjacencies(self, adjs: set[Adjacency]):
        for adj in adjs:
            neg_offset = adj.offset.negation()
            source_i = self.parts_to_index_mapping[adj.source]
            for neighbour in adj.allowed_neighbours:
                neighbour_i = self.parts_to_index_mapping[neighbour.other]
                self.ADJ[adj.offset][source_i, neighbour_i] = True
                self.ADJ_W[adj.offset][source_i, neighbour_i] = neighbour.weight
                if adj.symmetric:
                    self.ADJ[neg_offset][neighbour_i, source_i] = True
                    self.ADJ_W[neg_offset][neighbour_i, source_i] = neighbour.weight