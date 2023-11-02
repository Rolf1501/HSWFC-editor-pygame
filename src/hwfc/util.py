import numpy as np

# ====================================
# = Utility functions
# ====================================
def fancy_bmat(M):
    """
    Prints a numpy boolean array with fancy unicode squares.
    """
    try:
        f = lambda x : '◼' if x else '◻' 
        return np.vectorize(f)(M)
    except:
        return np.array([])

def clamp(n, smallest, largest): 
    """
    Clamps a number 'n' between the given minimum/maximum values.
    """
    return max(smallest, min(n, largest))

def color(r, g, b):
    """
    Converts a floating-point rgb color into a uint8 color (0.0-1.0 --> 0-255).
    """
    return (255 * np.array((r, g, b))).astype(int)
