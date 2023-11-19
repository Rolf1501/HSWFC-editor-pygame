from dataclasses import dataclass, field
from grid import Grid
from terminal import Terminal
from queue import Queue as Q
from coord import Coord


@dataclass
class Placement:
    terminal_id: int
    rotation: Coord
    # Up and orientation can be inferred from rotation and terminal_id
    up: Coord
    orientation: Coord
    


@dataclass
class WFC:
    grid: Grid
    terminals: list[Terminal]
    adjacencies: list
    seeds: list[Placement]
    propagation_queue: Q[int]


