import torch.nn as nn
from Enums import ActivationFunction

from Models.BaseCNN import BaseCNN

DEFAULT_NUM_FEATURES = 31
DEFAULT_NUM_CLASSES = 30
DEFAULT_DROPOUT = 0.0
DEFAULT_NUM_FILTERS = (64, 128)
DEFAULT_KERNEL_SIZE = 3
DEFAULT_POOL_SIZE = 2
DEFAULT_INPUT_CHANNELS = 1
DEFAULT_CLASSIFIER_HIDDEN_SIZE = 128


class KeystrokeCNN(BaseCNN):
	CONV = nn.Conv1d
	BN = nn.BatchNorm1d
	POOL = nn.MaxPool1d
	DROPOUT = nn.Dropout

	def __init__(
		self,
		num_features=DEFAULT_NUM_FEATURES,
		num_classes=DEFAULT_NUM_CLASSES,
		activation=ActivationFunction.RELU,
		dropout=DEFAULT_DROPOUT,
		num_filters=DEFAULT_NUM_FILTERS,
		kernel_size=DEFAULT_KERNEL_SIZE,
		pool_size=DEFAULT_POOL_SIZE,
		use_batch_norm=False,
	):
		super().__init__(
			num_classes=num_classes,
			input_channels=DEFAULT_INPUT_CHANNELS,
			num_filters=num_filters,
			kernel_size=kernel_size,
			pool_size=pool_size,
			activation=activation,
			dropout=dropout,
			use_batch_norm=use_batch_norm,
			classifier_hidden_size=DEFAULT_CLASSIFIER_HIDDEN_SIZE,
			input_spatial_size=num_features,
		)

	@staticmethod
	def _compute_flattened_size(input_spatial_size, pool_size, num_filters):
		sequence_length = input_spatial_size // (pool_size ** len(num_filters))
		return num_filters[-1] * sequence_length
