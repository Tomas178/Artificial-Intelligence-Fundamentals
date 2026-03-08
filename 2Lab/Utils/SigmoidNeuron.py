import time

import numpy as np
from numpy.typing import NDArray


class SigmoidNeuron:
	INTERVAL_START = -0.5
	INTERVAL_END = 0.5

	def __init__(self, n_features: int, learning_rate=0.1, epochs=500, e_min=0.001, seed=42):
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.e_min = e_min
		self.seed = seed

		# Pradiniai svoriai bei bias'ai iš intervalo [INTERVAL_START, INTERVAL_END]
		np.random.seed(seed)
		self.weights = np.random.uniform(self.INTERVAL_START, self.INTERVAL_END, n_features)
		self.bias = np.random.uniform(self.INTERVAL_START, self.INTERVAL_END)

		# Metrikų sėkimui
		self.train_errors = []
		self.validation_errors = []
		self.train_accuracies = []
		self.validation_accuracies = []
		self.training_time = 0.0
		self.epochs_run = 0

	def _sigmoid(self, a: NDArray[np.float64]) -> NDArray[np.float64]:
		return 1.0 / (1.0 + np.exp(-a))

	def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
		a = np.dot(X, self.weights) + self.bias
		return self._sigmoid(a)

	def classify(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
		return np.round(self.predict(X)).astype(int)

	def _compute_error(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> float:
		return np.sum((y_true, y_pred) ** 2)

	def _compute_accuracy(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> float:
		return np.mean(np.round(y_pred).astype(int) == y_true)

	def _reset_metrics(self):
		self.train_errors = []
		self.validation_errors = []

		self.train_accuracies = []
		self.validation_accuracies = []
