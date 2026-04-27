from enum import Enum


class Optimizer(Enum):
	ADAM = 'adam'
	SGD = 'sgd'
	RMSPROP = 'rmsprop'
	ADAMW = 'adamw'

	def __str__(self):
		return self.value
