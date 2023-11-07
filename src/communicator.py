class Communicator(object):
    def __init__(self) -> None:
        self.verbose = False

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Communicator, cls).__new__(cls)
        return cls.instance
    
    def communicate(self, message):
        print(message)