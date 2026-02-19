import numpy as np
from numpy.typing import NDArray


def step_activation(a: NDArray[np.float64]) -> NDArray[np.int_]:
	"""Returns an array of 1's & 0's"""
	return (a >= 0).astype(int)
