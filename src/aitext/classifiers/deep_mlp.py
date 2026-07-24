"""Deep MLP classifier (Strategy 2: PAWN features) -- the architecture that actually
produced the paper's PAWN+DeepMLP numbers in the "FinalResults*" notebooks:
Linear->BatchNorm->ReLU->Dropout stacked 3 times, trained for a fixed number of
epochs (no early stopping against the test set, unlike the exploratory grid-search
notebooks -- this is the methodologically cleaner variant, kept intentionally).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class DeepMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (1024, 512, 128),
        dropouts: tuple[float, ...] = (0.4, 0.3, 0.0),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim, dropout in zip(hidden_dims, dropouts):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMLPClassifier:
    """Thin sklearn-style wrapper (`.fit(X, y)` / `.predict_proba(X)`) around DeepMLP,
    so it can be used interchangeably with sklearn/xgboost classifiers by
    `aitext.classifiers.evaluation.holdout_eval`."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (1024, 512, 128),
        dropouts: tuple[float, ...] = (0.4, 0.3, 0.0),
        lr: float = 1e-5,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        batch_size: int = 32,
        seed: int = 42,
        device: str | None = None,
    ):
        self.hidden_dims = hidden_dims
        self.dropouts = dropouts
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: DeepMLP | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepMLPClassifier":
        torch.manual_seed(self.seed)
        self.model = DeepMLP(X.shape[1], self.hidden_dims, self.dropouts).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long()),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                criterion(self.model(xb), yb).backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Call .fit() before .predict_proba()."
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).float().to(self.device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs


def make_deep_mlp(seed: int = 42) -> DeepMLPClassifier:
    return DeepMLPClassifier(seed=seed)
