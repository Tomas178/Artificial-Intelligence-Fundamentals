import numpy as np


class DataSplitter:
	def __init__(self, features: np.ndarray, labels: np.ndarray):
		self.features = features
		self.labels = labels
		self.X_train = None
		self.y_train = None
		self.X_validation = None
		self.y_validation = None
		self.X_test = None
		self.y_test = None

	def split(self, train_ration=0.8):
		n = len(self.features)

		# Pirmasis padalijimas (80% mokymas+validavimas, 20% testavimas)
		first_split = int(n * train_ration)
		train_val_features = self.features[:first_split]
		train_val_labels = self.labels[:first_split]
		self.X_test = self.features[first_split:]
		self.y_test = self.labels[first_split:]

		# Antrasis padalijimas: 80 % mokymo 20 % validavimo (iš 80 % mokymo+validavmo aibės)
		second_split = int(len(train_val_features) * train_ration)
		self.X_train = train_val_features[:second_split]
		self.y_train = train_val_labels[:second_split]
		self.X_validation = train_val_features[second_split:]
		self.y_validation = train_val_labels[second_split:]

		print(f'Training set: {len(self.X_train)} samples')
		print(f'Validation set: {len(self.X_validation)} samples')
		print(f'Test set: {len(self.X_test)} samples')

		return (
			self.X_train,
			self.y_train,
			self.X_validation,
			self.y_validation,
			self.X_test,
			self.y_test,
		)
