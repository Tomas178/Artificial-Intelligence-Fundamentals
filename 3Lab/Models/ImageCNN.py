import torch.nn as nn
from consts import IMAGE_SIZE
from Enums import ActivationFunction

from Models.BaseCNN import BaseCNN

DEFAULT_NUM_CLASSES = 2
DEFAULT_DROPOUT = 0.0
DEFAULT_NUM_FILTERS = (32, 64, 128)
DEFAULT_KERNEL_SIZE = 3
DEFAULT_POOL_SIZE = 2
DEFAULT_INPUT_CHANNELS = 3
DEFAULT_IMAGE_SIZE = IMAGE_SIZE
DEFAULT_CLASSIFIER_HIDDEN_SIZE = 256


class ImageCNN(BaseCNN):
	CONV = nn.Conv2d
	BN = nn.BatchNorm2d
	POOL = nn.MaxPool2d
	DROPOUT = nn.Dropout2d

	def __init__(
		self,
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
			input_spatial_size=DEFAULT_IMAGE_SIZE,
		)

	@staticmethod
	def _compute_flattened_size(input_spatial_size, pool_size, num_filters):
		feature_map_size = input_spatial_size // (pool_size ** len(num_filters))
		return num_filters[-1] * feature_map_size * feature_map_size
