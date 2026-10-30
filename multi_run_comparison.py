from contextlib import redirect_stdout
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import torch

from nn import Network
from sine_wave_generator import SineWave


NUM_RUNS = 30
NUM_WAVES = 10
NUM_POINTS = 100
HIDDEN_SIZE = 20


def train_quietly(network, waves, reset_weights):
    """Train a network while keeping the console output manageable."""
    with redirect_stdout(StringIO()):
        if reset_weights:
            network.fit_reset_weights(waves)
        else:
            network.fit(waves)
    return np.array([losses[-1] for losses in network.losses_by_wave])


def run_pair(seed):
    """Compare both approaches on identical waves and task order."""
    rng = np.random.default_rng(seed)
    waves = [SineWave(NUM_POINTS, rng) for _ in range(NUM_WAVES)]
    order = rng.permutation(NUM_WAVES)
    ordered_waves = [waves[index] for index in order]

    torch.manual_seed(seed)
    inherited_network = Network(1, HIDDEN_SIZE, 1)
    torch.manual_seed(seed)
    reset_network = Network(1, HIDDEN_SIZE, 1)

    inherited_losses = train_quietly(
        inherited_network, ordered_waves, reset_weights=False
    )
    reset_losses = train_quietly(
        reset_network, ordered_waves, reset_weights=True
    )
    return inherited_losses, reset_losses


results = np.array([run_pair(seed) for seed in range(NUM_RUNS)])
inherited_losses = results[:, 0, :]
reset_losses = results[:, 1, :]
task_numbers = np.arange(1, NUM_WAVES + 1)

inherited_mean = inherited_losses.mean(axis=0)
reset_mean = reset_losses.mean(axis=0)
inherited_std = inherited_losses.std(axis=0)
reset_std = reset_losses.std(axis=0)

plt.figure(figsize=(10, 6))
plt.plot(
    task_numbers,
    inherited_mean,
    color="tab:red",
    linewidth=2,
    label="Inherited weights",
)
plt.fill_between(
    task_numbers,
    inherited_mean - inherited_std,
    inherited_mean + inherited_std,
    color="tab:red",
    alpha=0.2,
    label="Inherited weights: +/- 1 SD",
)
plt.plot(
    task_numbers,
    reset_mean,
    color="tab:blue",
    linewidth=2,
    label="Reset weights",
)
plt.fill_between(
    task_numbers,
    reset_mean - reset_std,
    reset_mean + reset_std,
    color="tab:blue",
    alpha=0.2,
    label="Reset weights: +/- 1 SD",
)

plt.xlabel("Task number")
plt.ylabel("Final training loss")
plt.title(f"Weight-reset comparison across {NUM_RUNS} runs")
plt.xticks(task_numbers)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Mean final loss, inherited weights: {inherited_mean.mean():.4f}")
print(f"Mean final loss, reset weights: {reset_mean.mean():.4f}")
if inherited_mean.mean() < reset_mean.mean():
    print("Lower average loss: inherited weights")
else:
    print("Lower average loss: reset weights")
