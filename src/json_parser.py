import json
import pathlib

from terminal import Terminal
from boundingbox import BoundingBox as BB
from util_data import Colour
import numpy as np

x0, y0, z0 = 2, 2, 2
mask0 = np.full((y0, x0, z0), True)
mask0[0, 0, 0] = False  # Create a small L shape
mask0[0, 1, 0] = False  # Create a small L shape
x1, y1, z1 = 1, 1, 1
mask1 = np.full((y1, x1, z1), True)

terminals = {
    0: Terminal(
        (x0, y0, z0),
        Colour(0.3, 0.6, 0.6, 1),
        mask=mask0,
    ),
    1: Terminal(
        (x1, y1, z1),
        Colour(0.8, 0.3, 0, 1),
        mask=mask1,
    ),
}
json_d = json.dumps(terminals[0].to_json())
d = json.loads(json_d)
print(Terminal(extent=d["extent"], colour=d["colour"], mask=np.asarray(d["mask"])))

curr_path = pathlib.Path(__file__).parent
json_folder = "json"
json_file = "json_test.json"
path = curr_path.joinpath(json_folder, json_file)
with open(path) as json_data:
    data = json.load(json_data)

    print(data["example"])
