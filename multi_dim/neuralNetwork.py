from torch import nn


class NeuralNetwork(nn.Module):
    """
    A 2-layers homogenous fully connected neural network from R^input_dim to R. The first layer has a bias term while
    the second layer doesn't.
    """
    def __init__(self, input_dim=2, hidden_layer_dim=1000):
        super().__init__()
        self.first_layer = nn.Linear(input_dim, hidden_layer_dim)
        self.first_activation = nn.ReLU()
        self.second_layer = nn.Linear(hidden_layer_dim, 1, bias=False)

    def forward(self, x):
        x = self.first_activation(self.first_layer(x))
        x = self.second_layer(x)
        return x
