from enum import Enum


class LossFunctions(Enum):
	CROSS_ENTROPY = 'cross_entropy'
	NLL = 'nll'

	def __str__(self):
		return self.value
