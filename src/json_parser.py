import json
from pathlib import Path

from terminal import Terminal
from boundingbox import BoundingBox as BB
from util_data import Colour
from coord import Coord
import numpy as np
from os import makedirs
from os.path import exists


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
json_d = json.dumps(terminals[0].to_json())
d = json.loads(json_d)
print(
    Terminal(
        extent=d["extent"],
        colour=d["colour"],
        mask=np.asarray(d["mask"]),
        distinct_orientations=d["distinct_orientations"],
        description=d["description"],
    )
)

curr_path = Path(__file__).parent
json_folder = "json"
json_file = "json_test.json"
path = curr_path.joinpath(json_folder, json_file)
with open(path) as json_data:
    data = json.load(json_data)

    print(data["example"])


class JSONParser:
    def __init__(self) -> None:
        self.terminal_file = "terminals.json"
        self.terminal_root = "terminals"
        self.adjacency_file = "adjacency.json"
        self.json_folder = "json"
        self.json_path = self.get_src_dir().joinpath(self.json_folder)
        self.terminal_path = self.json_path.joinpath(self.terminal_file)
        self.adjacency_path = self.json_path.joinpath(self.adjacency_file)
        self.init_file(self.terminal_path)

    def get_src_dir(self):
        return Path(__file__).parent

    def write_to_json(self, content, path: Path):
        with open(path, "w") as file:
            json.dump(content, file)

    def append_terminal(self, new_terminal, terminal_id):
        with open(self.terminal_path) as file:
            curr_data = json.load(file)
            curr_data[str(terminal_id)] = new_terminal
            self.write_to_json(curr_data, self.terminal_path)

    def init_file(self, path: Path):
        if not exists(path.parent):
            makedirs(path.parent)
        # Make sure the file can be loaded properly, otherwise reset.
        try:
            with open(path) as file:
                json.load(file)
        except:
            self.reset_file(path)

    def reset_file(self, path):
        with open(path, "w") as file:
            json.dump({}, file)

    def json_to_terminals(self):
        with open(self.terminal_path) as file:
            t_json = json.load(file)
            terminals = {}

            for t in t_json.keys():
                fields = Terminal.get_init_field_names()
                terminals[t] = Terminal(**{f: t_json[t][f] for f in fields})

            print(terminals)
            return terminals


jsonparser = JSONParser()

jsonparser.append_terminal(terminals[0].to_json(), 1)
ts = jsonparser.json_to_terminals()

for i in ts:
    print(ts[i])
