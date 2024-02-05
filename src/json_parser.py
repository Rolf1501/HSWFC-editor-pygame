import json
from os import makedirs
from os.path import exists
from pathlib import Path

from terminal import Terminal
from toy_examples import Example


class JSONParser:
    def __init__(self) -> None:
        self.terminal_file = "terminals.json"
        self.terminal_root = "terminals"
        self.examples_file = "examples.json"
        self.json_folder = "json"
        self.json_path = self.get_src_dir().joinpath(self.json_folder)
        self.terminal_path = self.json_path.joinpath(self.terminal_file)
        self.examples_path = self.json_path.joinpath(self.examples_file)
        self._init_file(self.terminal_path)

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

    def read_terminals(self):
        terminals = {}
        with open(self.terminal_path, "r") as file:
            t_json = json.load(file)
            terminals = {int(t): Terminal.from_json(t_json[t]) for t in t_json.keys()}

        return terminals

    def read_examples(self):
        examples = {}
        with open(self.examples_path, "r") as file:
            e_json = json.load(file)
            examples = {e: Example.from_json(e_json[e]) for e in e_json.keys()}
        return examples

    def write_terminals(self, terminals: list[Terminal]):
        for t in terminals:
            self.append_terminal(terminals[t], t)

    def write_example(self, example: Example):
        with open(self.examples_path) as file:
            curr_data = json.load(file)
            curr_data[example.name] = example.to_json()
            self.write_to_json(curr_data, self.examples_path)
