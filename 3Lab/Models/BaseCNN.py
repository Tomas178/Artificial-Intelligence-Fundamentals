import torch.nn as nn
from Enums import ActivationFunction

from Models.consts import ACTIVATION_LAYERS


class BaseCNN(nn.Module):
	CONV = None
	BN = None
	POOL = None
	DROPOUT = None

	def __init__(
		self,
		*,
		num_classes,
		input_channels,
		num_filters,
		kernel_size,
		pool_size,
		activation,
		dropout,
		use_batch_norm,
		classifier_hidden_size,
		input_spatial_size,
	):
		super().__init__()
		activation_layer = ACTIVATION_LAYERS[activation]

		convolutional_layers = []
		in_channels = input_channels

		for filter_count in num_filters:
			convolutional_layers.append(
				self.CONV(in_channels, filter_count, kernel_size, padding=1)
			)
			if use_batch_norm:
				convolutional_layers.append(self.BN(filter_count))
			convolutional_layers.append(activation_layer())
			convolutional_layers.append(self.POOL(pool_size))
			if dropout > 0:
				convolutional_layers.append(self.DROPOUT(dropout))
			in_channels = filter_count

		self.feature_extractor = nn.Sequential(*convolutional_layers)

		flattened_size = self._compute_flattened_size(input_spatial_size, pool_size, num_filters)

		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(flattened_size, classifier_hidden_size),
			activation_layer(),
			nn.Dropout(dropout),
			nn.Linear(classifier_hidden_size, num_classes),
		)

	@staticmethod
	def _compute_flattened_size(input_spatial_size, pool_size, num_filters):
		raise NotImplementedError

	def forward(self, x):
		x = self.feature_extractor(x)
		x = self.classifier(x)
		return x
