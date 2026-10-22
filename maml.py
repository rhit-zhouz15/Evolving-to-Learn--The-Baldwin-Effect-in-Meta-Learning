import torch
import torch.nn as nn
import numpy as np

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

    def forward(self, x):
        return self.network(x)

    def fit(self, wave_list):
        for i in range(len(wave_list)):
            wave = wave_list[i]
            x_cords = torch.tensor(np.concatenate([wave.x_cords]), dtype=torch.float32).unsqueeze(1)
            y_cords = torch.tensor(np.concatenate([wave.y_cords]), dtype=torch.float32).unsqueeze(1)

            for epoch in range(1500):
                self.optimizer.zero_grad()
                loss = self.loss(self.network(x_cords), y_cords)
                loss.backward()
                self.optimizer.step()
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                