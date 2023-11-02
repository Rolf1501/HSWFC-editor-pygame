import numpy as np

class Animator(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Animator, cls).__new__(cls)
        return cls.instance
  
    # offset = 0  # This global variable is used to determine when to reset frame recording
    def store_anim(self, iteration, i, j, c, reset=False):
        self.anim_frames, self.offset
        if reset:
            self.offset = len(self.anim_frames)
        # anim_frames.append([(i, j, c)])
        while (iteration + self.offset) >= len(self.anim_frames):
            self.anim_frames.append([])
        self.anim_frames[iteration + self.offset].append((i, j, np.array(c)))

    def set_offset(self, new_offset):
        self.offset = new_offset

    def set_record(self, new_record):
        self.record = new_record


