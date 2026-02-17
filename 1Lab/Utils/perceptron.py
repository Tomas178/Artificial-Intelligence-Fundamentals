import numpy as np
from Activation_functions.sigmoid_activation import sigmoid_activation
from Activation_functions.step_activation import step_activation
from Enums.Activation_function import ActivationFunction
from numpy.typing import NDArray


def perceptron(
	X: NDArray[np.floating], w1: float, w2: float, b: float, activation=ActivationFunction.STEP
) -> NDArray[np.int_]:
	# Apskaičiuojame a = x1*w1 + x2*w2 + b
	a = X[:, 0] * w1 + X[:, 1] * w2 + b

	if activation == ActivationFunction.STEP:
		predictions = step_activation(a)

	elif activation == ActivationFunction.SIGMOID:
		sig_values = sigmoid_activation(a)
		predictions = np.round(sig_values).astype(int)

	else:
		raise ValueError(f'Unknown activation function: {activation}')

	return predictions
