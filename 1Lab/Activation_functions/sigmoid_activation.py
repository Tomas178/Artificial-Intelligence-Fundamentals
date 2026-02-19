import numpy as np
from numpy.typing import NDArray


def sigmoid_activation(a: NDArray[np.float64]) -> NDArray[np.float64]:
	return 1.0 / (1.0 + np.exp(-a))
