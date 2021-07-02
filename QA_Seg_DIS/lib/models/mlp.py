from torch import nn
from lib.models.blocks import Dense



class MLP(nn.Module):
    def __init__(self, n_channels, dropout=0.):
        super().__init__()
        self.n_channels = n_channels

        hidden_layers = [Dense(in_channels, out_channels, "lrelu", dropout) for in_channels, out_channels in zip(n_channels[:-1], n_channels[1:])]
        self.hidden_layers = nn.Sequential(*hidden_layers)

    def forward(self, x):
        x = self.hidden_layers(x)

        return x