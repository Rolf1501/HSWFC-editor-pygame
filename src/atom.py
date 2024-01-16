from side_descriptor import SidesDescriptor


class Atom:
    def __init__(
        self, side_descriptor: SidesDescriptor, path="parts/1x1x1.glb"
    ) -> None:
        self.side_descriptor = side_descriptor  # Descriptor of what the open and filled faces of the atom are.
        self.path = path  # Path to the model of the atom.
        self.id = self.side_descriptor.to_int_id()  # Identifier of the atom.
