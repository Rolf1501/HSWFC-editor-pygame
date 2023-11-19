from dataclasses import dataclass, field
import numpy as np

@dataclass
class Grid:
    width: int
    height: int
    depth: int
    grid: np.ndarray = field(init=False)

    def __post_init__(self):
        self.grid = np.zeros((self.width, self.height, self.depth))

    def set_cell_value(self, value, x, y, z):
        self.grid[x, y, z] = value
