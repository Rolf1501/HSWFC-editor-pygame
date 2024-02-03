import json
import numpy as np
from os import makedirs
from os.path import exists
from pathlib import Path

from adjacencies import Adjacency, Relation as R
from coord import Coord
from dataclass_util import get_init_field_names
from terminal import Terminal
from util_data import Colour, Cardinals as C
from offsets import Offset


class JSONParser:
    def __init__(self) -> None:
        self.terminal_file = "terminals.json"
        self.terminal_root = "terminals"
        self.adjacency_file = "adjacency.json"
        self.json_folder = "json"
        self.json_path = self.get_src_dir().joinpath(self.json_folder)
        self.terminal_path = self.json_path.joinpath(self.terminal_file)
        self.adjacency_path = self.json_path.joinpath(self.adjacency_file)
        self._init_file(self.terminal_path)

    def get_src_dir(self):
        return Path(__file__).parent

    def write_to_json(self, content, path: Path):
        with open(path, "w") as file:
            json.dump(content, file, indent=2)

    def append_terminal(self, new_terminal: Terminal, terminal_id):
        with open(self.terminal_path) as file:
            curr_data = json.load(file)
            curr_data[str(terminal_id)] = new_terminal.to_json()
            self.write_to_json(curr_data, self.terminal_path)

    def append_adjacency(self, adjacency: Adjacency, example_name):
        with open(self.adjacency_path) as file:
            curr_data = json.load(file)
            if not example_name in curr_data or not isinstance(
                curr_data[example_name], list
            ):
                curr_data[example_name] = []
            curr_data[example_name].append(adjacency.to_json())
            self.write_to_json(curr_data, self.adjacency_path)

    def _init_file(self, path: Path):
        if not exists(path.parent):
            makedirs(path.parent)
        # Make sure the file can be loaded properly, otherwise reset.
        try:
            with open(path) as file:
                json.load(file)
        except:
            self._reset_file(path)

    def _reset_file(self, path):
        with open(path, "w") as file:
            json.dump({}, file)

    def read_terminals(self):
        terminals = {}
        with open(self.terminal_path) as file:
            t_json = json.load(file)

            fields = get_init_field_names(Terminal)
            for t in t_json.keys():
                # Translate the json data into terminal instances, following a dictionary approach.
                terminals[t] = Terminal(**{f: t_json[t][f] for f in fields})

        return terminals

    def read_adjacencies(self):
        adjacencies = []
        with open(self.adjacency_path) as file:
            a_json = json.load(file)
            fields = get_init_field_names(Adjacency)
            for example in a_json.keys():
                for a in a_json[example]:
                    adjacencies.append(Adjacency(**{f: a[f] for f in fields}))

        return adjacencies

    def write_terminals(self, terminals: list[Terminal]):
        for t in terminals:
            self.append_terminal(terminals[t], t)

    def write_adjacencies(self, adjacencies: list[Adjacency], example_name):
        for a in adjacencies:
            self.append_adjacency(a, example_name)


jsonparser = JSONParser()


x0, y0, z0 = 2, 2, 2
mask0 = np.full((y0, x0, z0), True)
mask0[0, 0, 0] = False  # Create a small L shape
mask0[0, 1, 0] = False  # Create a small L shape
x1, y1, z1 = 1, 1, 1
mask1 = np.full((y1, x1, z1), True)

terminals = {
    0: Terminal(
        Coord(x0, y0, z0),
        Colour(0.3, 0.6, 0.6, 1),
        mask=mask0,
    ),
    1: Terminal(
        Coord(x1, y1, z1),
        Colour(0.8, 0.3, 0, 1),
        mask=mask1,
    ),
}

adjacencies = [
    Adjacency(0, [R(0, 1)], Offset(*C.NORTH.value), True),
    Adjacency(0, [R(0, 1)], Offset(*C.EAST.value), True),
    Adjacency(0, [R(0, 1)], Offset(*C.SOUTH.value), True),
    Adjacency(0, [R(0, 1)], Offset(*C.WEST.value), True),
    Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.NORTH.value), True),
    Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.EAST.value), True),
    Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.SOUTH.value), True),
    Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.WEST.value), True),
    Adjacency(2, [R(0, 1), R(1, 1), R(2, 1)], Offset(*C.NORTH.value), True),
    Adjacency(2, [R(0, 1), R(1, 1), R(2, 1)], Offset(*C.EAST.value), True),
    Adjacency(2, [R(0, 1), R(1, 1), R(2, 1)], Offset(*C.SOUTH.value), True),
    Adjacency(2, [R(0, 1), R(1, 1), R(2, 1)], Offset(*C.WEST.value), True),
]


jsonparser._reset_file(jsonparser.terminal_path)
jsonparser._reset_file(jsonparser.adjacency_path)
jsonparser.write_terminals(terminals)
jsonparser.write_adjacencies(adjacencies, "test")

ts = jsonparser.read_terminals()
a = jsonparser.read_adjacencies()

for i in ts:
    print(i, ts[i])

for ai in a:
    print(ai)
