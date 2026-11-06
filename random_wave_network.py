import copy

import matplotlib.pyplot as plt
import numpy as np
import torch

from nn import Network
from sine_wave_generator import SineWave


NUM_WAVES = 3
NUM_POINTS_PER_WAVE = 100
HIDDEN_SIZE = 40
EPOCHS = 1000


def train_on_points(network, x_points, y_points, epochs, log_interval=None):
    """Train a network on one set of x/y points."""
    x_tensor = torch.tensor(x_points, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_points, dtype=torch.float32).unsqueeze(1)

    for epoch in range(epochs):
        network.optimizer.zero_grad()
        predictions = network(x_tensor)
        loss = network.loss(predictions, y_tensor)
        loss.backward()
        network.optimizer.step()

        should_log = log_interval and epoch % log_interval == 0
        if should_log or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


def predict(network, x_points):
    x_tensor = torch.tensor(x_points, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        return network(x_tensor).squeeze(1).numpy()


def train_each_wave(network, waves):
    """Train a fresh copy of the random network on each wave."""
    adapted_networks = []
    for index, wave in enumerate(waves, start=1):
        adapted_network = copy.deepcopy(network)
        train_on_points(
            adapted_network,
            wave.x_cords,
            wave.y_cords,
            EPOCHS,
        )
        adapted_networks.append(adapted_network)
        print(f"Finished training on wave {index}")
    return adapted_networks


def wave_mse(network, wave):
    predictions = predict(network, wave.x_cords)
    return np.mean((predictions - wave.y_cords) ** 2)


def plot_results(waves, random_networks):
    """Plot the random network's fit for each individual wave."""
    x_plot = np.linspace(0, 2 * np.pi, 500)
    figure, axes = plt.subplots(
        len(waves), 1, figsize=(8, 4 * len(waves)), squeeze=False
    )

    for index, (wave, random_network) in enumerate(
        zip(waves, random_networks)
    ):
        axis = axes[index, 0]
        axis.scatter(wave.x_cords, wave.y_cords, label="Wave points")
        axis.plot(
            x_plot,
            predict(random_network, x_plot),
            label="Random network",
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(f"Random network trained on wave {index + 1}")
        axis.legend()
        axis.grid(alpha=0.25)

    figure.tight_layout()
    plt.show()


def main():
    rng = np.random.default_rng(2)
    waves = [
        SineWave(NUM_POINTS_PER_WAVE, rng)
        for _ in range(NUM_WAVES)
    ]
    torch.manual_seed(1)
    initial_network = Network(1, HIDDEN_SIZE, 1)
    random_networks = train_each_wave(initial_network, waves)

    print("\nFinal training MSE by wave")
    for index, (random_network, wave) in enumerate(
        zip(random_networks, waves),
        start=1,
    ):
        print(f"Wave {index}: random={wave_mse(random_network, wave):.6f}")

    plot_results(waves, random_networks)


if __name__ == "__main__":
    main()
