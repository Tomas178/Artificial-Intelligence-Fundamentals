import numpy as np
from Activation_functions.sigmoid_activation import sigmoid_activation
from Activation_functions.step_activation import step_activation
from Enums.Activation_function import ActivationFunction
from numpy.typing import NDArray


def perceptron(
	X: NDArray[np.floating], w1: float, w2: float, w0: float, activation=ActivationFunction.STEP
) -> NDArray[np.int_]:
	"""Returns perceptron guesses that were calculated using activation function"""

	a = X[:, 0] * w1 + X[:, 1] * w2 + w0

	if activation == ActivationFunction.STEP:
		predictions = step_activation(a)

	elif activation == ActivationFunction.SIGMOID:
		sigmoid_values = sigmoid_activation(a)

		predictions = np.round(sigmoid_values).astype(int)

	else:
		raise ValueError(f'Unknown activation function: {activation}')

	return predictions
