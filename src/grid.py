from dataclasses import dataclass, field
import numpy as np

@dataclass
class AbstractGrid:
    width: int
    height: int
    depth: int
    default_fill_value: any = field(init=True, default=None)
    grid: np.ndarray = field(init=False)

    def __post_init__(self):
        pass

    def init_grid(self, value):
        self.grid = self.get_filled_grid(value)

    def get_filled_grid(self, default_value):
        return np.full((self.height, self.width, self.depth), default_value)
    
    def within_bounds(self, x, y, z):
        return x < self.width and y < self.height and z < self.depth and x >= 0 and y >= 0 and z >= 0
    
    def get(self, x, y, z):
        return self.grid[y,x,z]
    
    def set(self, x, y, z, value):
        self.grid[y,x,z] = value

class Grid(AbstractGrid):
    def __post_init__(self):
        self.init_grid(self.default_fill_value)

    def is_chosen(self, x, y, z) -> bool:
        return self.get(x,y,z) >= 0

@dataclass
class GridManager():
    width: int
    height: int
    depth: int
    grid: Grid = field(init=False)
    entropy: Grid = field(init=False)
    weighted_choices: Grid = field(init=False) # 4D array: 3D for cells, 1D for choices.
    debug_grid: Grid = field(init=False)

    def __post_init__(self):
        self.grid = Grid(self.width, self.height, self.depth, default_fill_value=-1)
        self.entropy = Grid(self.width, self.height, self.depth, default_fill_value=None)
        self.weighted_choices = Grid(self.width, self.height, self.depth, default_fill_value=None)
        self.debug_grid = Grid(self.width, self.height, self.depth, default_fill_value=None)

    def set_entropy(self, x, y, z, entropy):
        self.entropy.set(x,y,z, entropy)

    def init_w_choices(self, keys):
        for w in range(self.width):
            for h in range(self.height):
                for d in range(self.depth):
                    self.weighted_choices.set(w,h,d, np.asarray([[True, k, 1.0] for k in keys], dtype=float))
                    self.debug_grid.set(w,h,d, np.asarray([True for _ in keys], dtype=float))

    def init_entropy(self, max_entropy: float):
        self.entropy.init_grid(max_entropy)

    def set_w_choice(self, x, y, z, weighted_choices: np.ndarray):
        self.weighted_choices.set(x,y,z, weighted_choices)
