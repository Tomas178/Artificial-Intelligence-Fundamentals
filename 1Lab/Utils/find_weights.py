import numpy as np
from Enums.Activation_function import ActivationFunction
from numpy.typing import NDArray

from Utils.check_accuracy import check_accuracy

MAX_ITERATIONS_COUNT = 1_000_000
COLLECTIONS_COUNT = 3
DEFAULT_SEARCH_RANGE = (-10.0, 10.0)


def find_weights(
	X: NDArray[np.floating],
	y: NDArray[np.int_],
	activation: ActivationFunction,
	count=COLLECTIONS_COUNT,
	search_range: tuple[float, float] = DEFAULT_SEARCH_RANGE,
	max_iterations=MAX_ITERATIONS_COUNT,
) -> list[tuple[float, float, float]]:
	"""Returns the 'count' sets of (w1, w2, w0)"""
	found: list[tuple[float, float, float]] = []
	iterations: int = 0

	while len(found) < count and iterations < max_iterations:
		w1 = np.random.uniform(*search_range)
		w2 = np.random.uniform(*search_range)
		w0 = np.random.uniform(*search_range)
		iterations += 1

		if check_accuracy(X, y, w1, w2, w0, activation):
			found.append((w1, w2, w0))
			print(
				f'Rinkinys {len(found)}: '
				f'w1={w1:.4f}, w2={w2:.4f}, w0={w0:.4f}  '
				f'(rastas po {iterations} iteracijų)'
			)

	print(f'\nIš viso patikrinta kombinacijų: {iterations}')
	return found
