import numpy as np
from Enums.Activation_function import ActivationFunction
from numpy.typing import NDArray

from Utils.check_accuracy import check_accuracy

MAX_ITERATIONS_COUNT = 1_000_000


def find_weights(
	X: NDArray[np.floating],
	y: NDArray[np.int_],
	activation: ActivationFunction,
	count: int = 3,
	search_range: tuple[float, float] = (-10.0, 10.0),
	max_iterations: int = MAX_ITERATIONS_COUNT,
) -> list[tuple[float, float, float]]:
	found: list[tuple[float, float, float]] = []
	iterations: int = 0

	while len(found) < count and iterations < max_iterations:
		w1: float = np.random.uniform(*search_range)
		w2: float = np.random.uniform(*search_range)
		b: float = np.random.uniform(*search_range)
		iterations += 1

		if check_accuracy(X, y, w1, w2, b, activation):
			found.append((w1, w2, b))
			print(
				f'Rinkinys {len(found)}: '
				f'w1={w1:.4f}, w2={w2:.4f}, b={b:.4f}  '
				f'(rastas po {iterations} iteracijų)'
			)

	print(f'\nIš viso patikrinta kombinacijų: {iterations}')
	return found
