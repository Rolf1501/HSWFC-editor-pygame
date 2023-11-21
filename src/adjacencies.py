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
    adjacencies: list[Adjacency] = field(default_factory=None)
    offsets: np.matrix = field(default_factory=OffsetFactory().get_offsets(dimensions=3, cardinal=True))
    ADJ: dict[Offset, np.ndarray] = field(init=False)
    parts_to_index_mapping: bidict[int, int] = field(init=False)

    def __post_init__(self):
        self.parts_to_index_mapping = bidict({self.parts[i]: i for i in range(len(self.parts))})
        for offset in self.offsets:
            self.ADJ[offset] = {p: np.full((len(self.parts)), False) for p in self.parts}


    def allow_adjacencies(self, adjs: set[Adjacency]):
        for adj in adjs:
            neg_offset = adj.offset.negation()
            source_i = self.parts_to_index_mapping[adj.source]
            for other in adj.allowed_neighbours:
                other_i = self.parts_to_index_mapping[other]
                self.ADJ[adj.offset][source_i, other_i] = True
                if adj.symmetric:
                    self.ADJ[neg_offset][other_i, source_i] = True