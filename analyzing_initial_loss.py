from nn import Network
from sine_wave_generator import SineWave, plot_sine_wave, pool_sine_wave_points
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset

def to_dataset(wave_list):
    x, y = pool_sine_wave_points(wave_list)
    return TensorDataset(torch.tensor(x, dtype=torch.float32).unsqueeze(1),
                         torch.tensor(y, dtype=torch.float32).unsqueeze(1))

num_points = 10
hidden_size = 40
rng = np.random.default_rng(2)
torch.manual_seed(1)

wave_list = [SineWave(num_points, rng) for _ in range(3)]
plot_sine_wave(wave_list[0])
plot_sine_wave(wave_list[1])
plot_sine_wave(wave_list[2])

networks = [Network(1, hidden_size, 1) for _ in range(100)]
initial_losses_task_1 = []
initial_losses_task_2 = []
initial_losses_task_3 = []

for network in networks:
    network.fit(wave_list)
    initial_losses_task_1.append(network.losses_by_wave[0][0])
    initial_losses_task_2.append(network.losses_by_wave[1][0])
    initial_losses_task_3.append(network.losses_by_wave[2][0])

plt.plot(initial_losses_task_1, color='blue', label="Initial loss on task 1")
plt.plot(initial_losses_task_2, color='red', label="Initial loss on task 2")
plt.plot(initial_losses_task_3, color='green', label="Initial loss on task 3")

plt.legend()
plt.show()

network_indices = np.arange(len(networks))

plt.scatter(initial_losses_task_1, network_indices,
            color='blue', label="Initial loss on task 1")
plt.scatter(initial_losses_task_2, network_indices,
            color='red', label="Initial loss on task 2")
plt.scatter(initial_losses_task_3, network_indices,
            color='green', label="Initial loss on task 3")

plt.xlabel("Initial loss")
plt.ylabel("Network index")

plt.legend()
plt.show()