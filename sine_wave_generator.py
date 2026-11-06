import numpy as np
import random
import matplotlib.pyplot as plt

class SineWave():
    def __init__(self, num_points, rng):
        self.a = rng.uniform(0.1, 3)
        self.b = rng.uniform(0.1, 3)
        self.c = rng.uniform(0, 3)
        # self.d = rng.uniform(0, 3)
        self.x_cords = np.linspace(0, 2*np.pi, num_points)
        self.y_cords = self.a * np.sin(self.b * (self.x_cords - self.c))

def pool_sine_wave_points(wave_list):
    """Return all x and y coordinates from a collection of sine waves."""
    x_cords = np.concatenate([wave.x_cords for wave in wave_list])
    y_cords = np.concatenate([wave.y_cords for wave in wave_list])
    return x_cords, y_cords

def plot_sine_wave(wave):
    plt.plot(wave.x_cords, wave.y_cords)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Generated sine wave")
    plt.show()
