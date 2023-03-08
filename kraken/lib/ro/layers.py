"""
Layers for VGSL models
"""
from torch import nn

# all tensors are ordered NCHW, the "feature" dimension is C, so the output of
# an LSTM will be put into C same as the filters of a CNN.

__all__ = ['MLP']


class MLP(nn.Module):
    """
    A simple 2 layer MLP for reading order determination.
    """
    def __init__(self, feature_size: int, hidden_size: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
