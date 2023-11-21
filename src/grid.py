from dataclasses import dataclass, field
import numpy as np
from terminal import Terminal


@dataclass
class AbstractGrid:
    width: int
    height: int
    depth: int
    grid: np.ndarray = field(init=False)

    def __post_init__(self):
        pass

    def set_cell_value(self, value, x, y, z):
        self.grid[x, y, z] = value

    def init_grid(self, value):
        self.grid = self.get_filled_grid(value)

    def get_filled_grid(self, default_value):
        return np.full((self.width, self.height, self.depth), default_value)

class Grid(AbstractGrid):
    def __post_init__(self):
        self.init_grid(-1)

class GridInfo(AbstractGrid):
    parts: list[Terminal]
    grid: np.ndarray = field(init=False)
    entropy: np.ndarray = field(init=False)
    weighted_choices: np.ndarray = field(init=False)

    def __post_init__(self):
        self.grid = self.get_filled_grid(None)
        self.entropy = self.get_filled_grid(None)
        self.weighted_choices = self.get_filled_grid(None)


        
    
