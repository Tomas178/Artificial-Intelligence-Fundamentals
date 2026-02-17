import numpy as np


def sigmoid_activation(a: float) -> float:
	return 1.0 / (1.0 + np.exp(-a))
