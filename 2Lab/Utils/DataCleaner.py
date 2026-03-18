import numpy as np


class DataCleaner:
	RANDOM_SEED = 42

	def __init__(self, filepath: str):
		self.filepath = filepath
		self.features = None
		self.labels = None

	def clean(self):
		data = []
		with open(self.filepath, 'r') as f:
			for line in f:
				if not line:
					continue

				parts = line.split(',')
				if '?' in parts:
					continue

				data.append(parts)

		data = np.array(data, dtype=float)

		print(f'Total records after removing missing values: {len(data)}')

		# Pašaliname ID stulpelį
		self.features = data[:, 1:10]
		self.labels = data[:, 10]

		# Konvertuojame 2 ir 4 klases atitinkamai į 0 ir 1 klases
		self.labels = np.where(self.labels == 2, 0, 1)

		print(f'Total features: {self.features.shape[1]}')
		print(f'Class 0 (benign): {np.sum(self.labels == 0)}')
		print(f'Class 1 (malignant): {np.sum(self.labels == 1)}')

		# Išmaišome
		np.random.seed(self.RANDOM_SEED)
		indices = np.random.permutation(len(self.features))
		self.features = self.features[indices]
		self.labels = self.labels[indices]

		# Normalizuojame iki [0, 1] min-max metodu
		feature_min = self.features.min(axis=0)
		feature_max = self.features.max(axis=0)
		self.features = (self.features - feature_min) / (feature_max - feature_min + 1e-8)

		return self.features, self.labels

	def save(self, output_filepath: str):
		if self.features is None or self.labels is None:
			raise ValueError('Data has not been cleaned yet. Call clean() first.')

		combined = np.column_stack([self.features, self.labels])
		np.savetxt(output_filepath, combined, delimiter=',', fmt='%.8f')

		print(f'Cleaned data saved to {output_filepath}')
