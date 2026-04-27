from enum import Enum


class ActivationFunction(Enum):
	RELU = 'relu'
	LEAKY_RELU = 'leaky_relu'
	TANH = 'tanh'
	SIGMOID = 'sigmoid'
	ELU = 'elu'

	def __str__(self):
		return self.value
