from enum import Enum
class Verbosity(Enum):
    LOW = 0
    MID = 1
    HIGH = 2

class Communicator(object):
    def __init__(self, verbosity: Verbosity=Verbosity.HIGH):
        self.verbosity = verbosity

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Communicator, cls).__new__(cls)
        return cls.instance
    
    def communicate(self, message, verbosity: Verbosity = Verbosity.HIGH):
        if verbosity.value >= self.verbosity.value:
            print(message)