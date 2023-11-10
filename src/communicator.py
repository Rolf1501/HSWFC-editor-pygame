from enum import Enum
class Verbosity(Enum):
    LOW = 2
    MID = 1
    HIGH = 0

class Communicator(object):
    def __init__(self, verbosity: Verbosity=Verbosity.LOW):
        self.verbosity = verbosity

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Communicator, cls).__new__(cls)
        return cls.instance
    
    def communicate(self, message, verbosity: Verbosity = Verbosity.LOW):
        if verbosity.value >= self.verbosity.value:
            print(message)
    
    def set_verbosity(self, v: Verbosity):
        self.verbosity = v
    
    def cycle_verbosity(self, increase=True):
        diff = 1 if increase else -1
        self.set_verbosity(Verbosity((self.verbosity.value + diff) % len(Verbosity)))