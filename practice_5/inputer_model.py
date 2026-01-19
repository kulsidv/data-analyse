import torch
import numpy as np


class ThreeLayersNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(9, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)

    def fit(self, X_train, y_train, loss=None, optimizer=None, epoches=1000):
        if not loss:
            loss = torch.nn.MSELoss()
        if not optimizer:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        for _ in range(epoches):
            optimizer.zero_grad()
            logits = self(torch.tensor(X_train, dtype=torch.float32))
            loss_value = loss(logits, torch.tensor(y_train, dtype=torch.float32))
            loss_value.backward()
            optimizer.step()

    def predict(self, X_test, y_test=None, bench=False):
        self.eval()
        with torch.no_grad():
            X = torch.tensor(X_test, dtype=torch.float32)
            preds = self(X).numpy()

        if bench:
            errors = np.mean(np.abs(preds - y_test))
            return preds, errors
        return preds, None
