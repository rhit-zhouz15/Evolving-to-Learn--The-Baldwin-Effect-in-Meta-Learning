import copy

import matplotlib.pyplot as plt
import numpy as np
import torch

from nn import Network
from sine_wave_generator import SineWave, pool_sine_wave_points


NUM_WAVES = 3
NUM_POINTS_PER_WAVE = 100
HIDDEN_SIZE = 40
EPOCHS = 50000
ADAPTATION_EPOCHS = 100


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


def plot_pooled_result(network, x_points, y_points):
    """Plot pooled training points and the network's fitted output."""
    x_plot = np.linspace(x_points.min(), x_points.max(), 500)
    predictions = predict(network, x_plot)

    plt.scatter(x_points, y_points, alpha=0.45, label="Pooled wave points")
    plt.plot(x_plot, predictions, color="black", linewidth=2,
             label="Network output")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Network trained on pooled sine-wave points")
    plt.legend()
    plt.tight_layout()
    plt.show()


def adapt_to_waves(network, waves):
    """Copy the pooled model and fine-tune one copy independently per wave."""
    adapted_networks = []
    for index, wave in enumerate(waves, start=1):
        adapted_network = copy.deepcopy(network)
        train_on_points(
            adapted_network,
            wave.x_cords,
            wave.y_cords,
            ADAPTATION_EPOCHS,
        )
        adapted_networks.append(adapted_network)
        print(f"Finished adaptation on wave {index}")
    return adapted_networks


def plot_adapted_results(waves, pooled_network, adapted_networks):
    """Plot each wave with its pooled and individually adapted predictions."""
    x_plot = np.linspace(0, 2 * np.pi, 500)
    figure, axes = plt.subplots(
        len(waves), 1, figsize=(8, 4 * len(waves)), squeeze=False
    )

    for index, (wave, adapted_network) in enumerate(
        zip(waves, adapted_networks)
    ):
        axis = axes[index, 0]
        axis.scatter(wave.x_cords, wave.y_cords, label="Wave points")
        axis.plot(
            x_plot,
            predict(pooled_network, x_plot),
            "--",
            label="Pooled model",
        )
        axis.plot(
            x_plot,
            predict(adapted_network, x_plot),
            label="After individual adaptation",
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(f"Adaptation on wave {index + 1}")
        axis.legend()
        axis.grid(alpha=0.25)

    figure.tight_layout()
    plt.show()


def main():
    rng = np.random.default_rng(55)
    waves = [
        SineWave(NUM_POINTS_PER_WAVE, rng)
        for _ in range(NUM_WAVES)
    ]
    x_points, y_points = pool_sine_wave_points(waves)

    torch.manual_seed(1)
    network = Network(1, HIDDEN_SIZE, 1)
    train_on_points(network, x_points, y_points, EPOCHS, log_interval=100)
    plot_pooled_result(network, x_points, y_points)
    adapted_networks = adapt_to_waves(network, waves)
    plot_adapted_results(waves, network, adapted_networks)


if __name__ == "__main__":
    main()
