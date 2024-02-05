from abc import ABC, abstractmethod


class JSONMeta(ABC):
    @abstractmethod
    def to_json(self):
        pass

    @classmethod
    @abstractmethod
    def from_json(self):
        pass
