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
        return (
            x < self.width
            and y < self.height
            and z < self.depth
            and x >= 0
            and y >= 0
            and z >= 0
        )

    def get(self, x, y, z):
        return (
            self.grid[int(y), int(x), int(z)] if self.within_bounds(x, y, z) else None
        )

    def set(self, x, y, z, value):
        if self.within_bounds(x, y, z):
            self.grid[int(y), int(x), int(z)] = value

    def print_xy(self):
        for z in range(self.depth):
            print(self.grid[:, :, z])

    def print_xz(self):
        for y in range(self.height):
            print(np.rot90(self.grid[y, :, :]))

    def print_yz(self):
        for x in range(self.width):
            print(self.grid[:, x, :])


class Grid(AbstractGrid):
    def __post_init__(self):
        self.init_grid(self.default_fill_value)

    def is_chosen(self, x, y, z) -> bool:
        choice = self.get(x, y, z)
        if self.default_fill_value is not None:
            return choice > self.default_fill_value
        else:
            return choice is not None


@dataclass
class GridManager:
    width: int
    height: int
    depth: int
    grid: Grid = field(init=False)
    entropy: Grid = field(init=False)
    weighted_choices: Grid = field(
        init=False
    )  # 4D array: 3D for cells, 1D for choices.
    choice_booleans: Grid = field(init=False)
    choice_ids: Grid = field(init=False)
    choice_weights: Grid = field(init=False)

    def __post_init__(self):
        self.grid = Grid(self.width, self.height, self.depth, default_fill_value=None)
        self.entropy = Grid(
            self.width, self.height, self.depth, default_fill_value=None
        )
        self.choice_booleans = Grid(
            self.width, self.height, self.depth, default_fill_value=None
        )
        self.choice_ids = Grid(
            self.width, self.height, self.depth, default_fill_value=None
        )
        self.choice_weights = Grid(
            self.width, self.height, self.depth, default_fill_value=None
        )

    def set_entropy(self, x, y, z, entropy):
        self.entropy.set(x, y, z, entropy)

    def init_w_choices(self, default_weights):
        # default_choices = [[True, k, default_weights[k]] for k in default_weights]
        default_choice_b = [True for _ in default_weights]
        default_choice_id = [k for k in default_weights]
        default_choice_w = [default_weights[k] for k in default_weights]
        for w in range(self.width):
            for h in range(self.height):
                for d in range(self.depth):
                    self.choice_booleans.set(
                        w, h, d, np.asarray(default_choice_b, dtype=bool)
                    )
                    self.choice_ids.set(
                        w, h, d, np.asarray(default_choice_id, dtype=float)
                    )
                    self.choice_weights.set(
                        w, h, d, np.asarray(default_choice_w, dtype=float)
                    )

    def init_entropy(self, max_entropy: float):
        self.entropy.init_grid(max_entropy)
