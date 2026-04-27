import torch.nn as nn
from Enums import ActivationFunction

ACTIVATION_LAYERS = {
	ActivationFunction.RELU: nn.ReLU,
	ActivationFunction.LEAKY_RELU: nn.LeakyReLU,
	ActivationFunction.TANH: nn.Tanh,
	ActivationFunction.SIGMOID: nn.Sigmoid,
	ActivationFunction.ELU: nn.ELU,
}
