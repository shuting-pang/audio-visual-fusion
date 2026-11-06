"""Small binary classifier for the feature-level Filter-and-Concat example."""

from torch import nn


class ConcatClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 2),
        )

    def forward(self, fused_features):
        """Return [N, 2] logits; apply cross-entropy against binary labels."""
        return self.network(fused_features)
