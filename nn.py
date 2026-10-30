import torch
import torch.nn as nn
import numpy as np
import copy

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.initial_params = copy.deepcopy(self.network.state_dict())
        self.losses_by_wave = []

    def forward(self, x):
        return self.network(x)

    def fit(self, wave_list):
        for i in range(len(wave_list)):
            wave = wave_list[i]
            x_cords = torch.tensor(np.concatenate([wave.x_cords]), dtype=torch.float32).unsqueeze(1)
            y_cords = torch.tensor(np.concatenate([wave.y_cords]), dtype=torch.float32).unsqueeze(1)
            losses = []

            for epoch in range(100):
                self.optimizer.zero_grad()
                loss = self.loss(self.network(x_cords), y_cords)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            self.losses_by_wave.append(losses)

    def fit_reset_weights(self, wave_list):
            for i in range(len(wave_list)):
                self.reset_all()
                wave = wave_list[i]
                x_cords = torch.tensor(np.concatenate([wave.x_cords]), dtype=torch.float32).unsqueeze(1)
                y_cords = torch.tensor(np.concatenate([wave.y_cords]), dtype=torch.float32).unsqueeze(1)
                losses = []
    
                for epoch in range(100):
                    self.optimizer.zero_grad()
                    loss = self.loss(self.network(x_cords), y_cords)
                    losses.append(loss.item())
                    loss.backward()
                    self.optimizer.step()
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                self.losses_by_wave.append(losses)

    def reset_losses(self):
        self.losses_by_wave = []

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def reset_all(self):
        self.network.load_state_dict(copy.deepcopy(self.initial_params))
        self.reset_optimizer()
                