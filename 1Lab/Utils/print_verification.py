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
	print('\n' + '=' * 60)
	print('PATIKRINIMAS')
	print('=' * 60)

	print('\nSlenkstinė aktyvacijos funkcija:')
	for i, (w1, w2, b) in enumerate(found_step):
		preds: NDArray[np.int_] = perceptron(X, w1, w2, b, ActivationFunction.STEP)
		acc: float = float(np.mean(preds == y)) * 100
		print(f'  Rinkinys {i + 1}: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f} → tikslumas: {acc:.1f}%')

	print('\nSigmoidinė aktyvacijos funkcija:')
	for i, (w1, w2, b) in enumerate(found_sigmoid):
		preds = perceptron(X, w1, w2, b, ActivationFunction.SIGMOID)
		acc = float(np.mean(preds == y)) * 100
		print(f'  Rinkinys {i + 1}: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f} → tikslumas: {acc:.1f}%')
