from dataclasses import dataclass, field
from grid import Grid, GridInfo
from terminal import Terminal
from queue import Queue as Q
from coord import Coord
from util_data import Cardinals
from adjacencies import AdjacencyMatrix, Adjacency

@dataclass
class Placement:
    terminal_id: int
    rotation: Coord
    # Up and orientation can be inferred from rotation and terminal_id
    up: Cardinals
    orientation: Cardinals
    

@dataclass
class WFC:
    terminals: list[Terminal]
    adjacencies: list[Adjacency]
    grid_extent: Coord = field(default=Coord(10,10,10))
    seeds: list[Placement] = field(default=None)
    grid: Grid = field(init=False)
    grid_info: GridInfo = field(init=False)
    propagation_queue: Q[int] =field(init=False)
    adj_matrix: AdjacencyMatrix = field(init=False)


