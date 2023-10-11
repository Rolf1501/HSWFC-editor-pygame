from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Input:
    """
    This dataclass represents an input. Mostly made to easily perform equals between the images during input reading, in order to eliminate duplicates.
    """
    img: any
    mask: any
    
    def __eq__(self, other):
        if not isinstance(other, Input):
            return NotImplemented
        return np.all(self.img==other.img) and np.all(self.mask==other.mask)
    
    def __hash__(self):
        return hash(str(self.img)+str(self.mask))