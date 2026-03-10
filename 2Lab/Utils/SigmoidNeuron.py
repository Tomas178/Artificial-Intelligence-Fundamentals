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

		np.random.seed(seed)
		self.weights = np.random.uniform(self.INTERVAL_START, self.INTERVAL_END, n_features)
		self.bias = np.random.uniform(self.INTERVAL_START, self.INTERVAL_END)

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

	def classify(self, X: np.ndarray) -> np.ndarray:
		return np.round(self.predict(X)).astype(int)

	def _compute_error(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
		return np.sum((y_true - y_pred) ** 2) / len(y_true)

	def _compute_accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
		return np.mean(np.round(y_pred).astype(int) == y_true)

	def _reset_metrics(self):
		self.train_errors = []
		self.validation_errors = []
		self.train_accuracies = []
		self.validation_accuracies = []

	def _record_metrics(self, X_train, y_train, X_val, y_val, train_error):
		self.train_errors.append(train_error)
		self.train_accuracies.append(self._compute_accuracy(self.predict(X_train), y_train))

		y_pred_val = self.predict(X_val)
		self.validation_errors.append(self._compute_error(y_pred_val, y_val))
		self.validation_accuracies.append(self._compute_accuracy(y_pred_val, y_val))

	def _train(self, X_train, y_train, X_val, y_val, stochastic=False):
		self._reset_metrics()
		start_time = time.time()

		total_error = float('inf')
		epoch = 0

		while total_error > self.e_min and epoch < self.epochs:
			indices = np.random.permutation(len(X_train))
			X_shuffled = X_train[indices]
			y_shuffled = y_train[indices]

			total_error = 0.0
			gradient_sum_w = np.zeros(len(self.weights))
			gradient_sum_b = 0.0

			for i in range(len(X_shuffled)):
				x_i = X_shuffled[i]
				t_i = y_shuffled[i]
				y_i = self.predict(x_i.reshape(1, -1))[0]

				grad = (y_i - t_i) * y_i * (1 - y_i)

				if stochastic:
					self.weights -= self.learning_rate * grad * x_i
					self.bias -= self.learning_rate * grad
				else:
					gradient_sum_w += grad * x_i
					gradient_sum_b += grad

				total_error += (t_i - y_i) ** 2

			m = len(X_shuffled)
			total_error = total_error / m

			if not stochastic:
				self.weights -= self.learning_rate * (gradient_sum_w / m)
				self.bias -= self.learning_rate * (gradient_sum_b / m)

			self._record_metrics(X_train, y_train, X_val, y_val, total_error)
			epoch += 1

		self.epochs_run = epoch
		self.training_time = time.time() - start_time

	# Paketinis gradientinis nusileidimas
	def train_batch(self, X_train, y_train, X_val, y_val):
		self._train(X_train, y_train, X_val, y_val, stochastic=False)

	# Stochastinis gradientinis nusileidimas
	def train_stochastic(self, X_train, y_train, X_val, y_val):
		self._train(X_train, y_train, X_val, y_val, stochastic=True)

	def evaluate(self, X: np.ndarray, y: np.ndarray):
		y_pred = self.predict(X)
		error = self._compute_error(y_pred, y)
		accuracy = self._compute_accuracy(y_pred, y)
		predictions = self.classify(X)
		return error, accuracy, predictions, y_pred
