from nn import Network
from sine_wave_generator import SineWave
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset

def to_dataset(wave_list):
    x = np.concatenate([wave.x_cords for wave in wave_list])
    y = np.concatenate([wave.y_cords for wave in wave_list])
    return TensorDataset(torch.tensor(x, dtype=torch.float32).unsqueeze(1),
                         torch.tensor(y, dtype=torch.float32).unsqueeze(1))

num_points = 100
hidden_size = 20
rng = np.random.default_rng(1)

wave_list = [SineWave(num_points, rng) for _ in range(10)]

n_train = int(0.7 * len(wave_list))
rand_idx = rng.permutation(len(wave_list))
train_waves = [wave_list[i] for i in rand_idx[:n_train]]
test_waves = [wave_list[i] for i in rand_idx[n_train:]]

network = Network(1, hidden_size, 1)

for i in range(len(train_waves)):
    print(f"Training on task {i}")
    network.fit(train_waves[i])
    y_predictions = []

    with torch.no_grad():
        x = torch.tensor(train_waves[i].x_cords, dtype=torch.float32).unsqueeze(1)
        y_predictions = network(x).squeeze(1).numpy()
        
    xpoints = np.array(train_waves[i].x_cords)
    ypoints = np.array(train_waves[i].y_cords)

    plt.plot(xpoints, ypoints)
    plt.plot(xpoints, y_predictions)
    plt.show()


wave = test_waves[1]
y_predictions = []

with torch.no_grad():
    x = torch.tensor(wave.x_cords, dtype=torch.float32).unsqueeze(1)
    y_predictions = network(x).squeeze(1).numpy()

xpoints = np.array(wave.x_cords)
ypoints = np.array(wave.y_cords)

plt.plot(xpoints, ypoints)
plt.plot(xpoints, y_predictions)
plt.show()