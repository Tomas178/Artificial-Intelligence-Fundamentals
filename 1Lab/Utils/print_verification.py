import numpy as np
from Enums.Activation_function import ActivationFunction
from numpy.typing import NDArray

from Utils.perceptron import perceptron


def print_verification(
	X: NDArray[np.floating],
	y: NDArray[np.int_],
	found_step: list[tuple[float, float, float]],
	found_sigmoid: list[tuple[float, float, float]],
) -> None:
	print('PATIKRINIMAS')

	print('\nSlenkstinė aktyvacijos funkcija:')
	for i, (w1, w2, b) in enumerate(found_step):
		predictions = perceptron(X, w1, w2, b, ActivationFunction.STEP)
		accuracy = float(np.mean(predictions == y)) * 100
		print(f'Rinkinys {i + 1}: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f} → tikslumas: {accuracy:.1f}%')

	print('\nSigmoidinė aktyvacijos funkcija:')
	for i, (w1, w2, b) in enumerate(found_sigmoid):
		predictions = perceptron(X, w1, w2, b, ActivationFunction.SIGMOID)
		accuracy = float(np.mean(predictions == y)) * 100
		print(f'Rinkinys {i + 1}: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f} → tikslumas: {accuracy:.1f}%')
