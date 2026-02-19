import numpy as np
from Enums.Activation_function import ActivationFunction
from numpy.typing import NDArray

from Utils.perceptron import perceptron


def check_accuracy(
	X: NDArray[np.floating],
	y: NDArray[np.int_],
	w1: float,
	w2: float,
	b: float,
	activation: ActivationFunction = ActivationFunction.STEP,
) -> bool:
	"""Checks if all prediction by perceptron are correct and returns either true or false."""

	predictions = perceptron(X, w1, w2, b, activation)
	return bool(np.all(predictions == y))
