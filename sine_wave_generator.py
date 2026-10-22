import numpy as np
import random

class SineWave():
    def __init__(self, num_points, rng):
        self.a = rng.uniform(0.1, 3)
        self.b = rng.uniform(0.1, 3)
        self.c = rng.uniform(0, 3)
        self.d = rng.uniform(0, 3)
        self.x_cords = np.linspace(0, 2*np.pi, num_points)
        self.y_cords = self.a * np.sin(self.b * (self.x_cords - self.c)) + self.d

    
