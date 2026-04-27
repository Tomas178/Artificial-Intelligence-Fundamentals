import os

import pandas as pd
from consts import FILENAME_TEST, FILENAME_TRAIN, FILENAME_VALIDATION
from sklearn.model_selection import train_test_split

from DataSplitter import BaseDataSplitter
from DataSplitter.consts import RANDOM_STATE, SPLIT_SIZE

COLUMN_SUBJECT = 'subject'
COLUMN_SESSION_INDEX = 'sessionIndex'
COLUMNS_TO_DROP = [COLUMN_SUBJECT, COLUMN_SESSION_INDEX, 'rep']
EPSILON = 1e-8

SUBJECTS_COUNT = 30
SESSION_COUNT = 1


class KeystrokeDataSplitter(BaseDataSplitter):
	def __init__(
		self,
		csv_path,
		output_dir,
		n_subjects=SUBJECTS_COUNT,
		session=SESSION_COUNT,
		random_state=RANDOM_STATE,
	):
		self.csv_path = csv_path
		self.output_dir = output_dir
		self.n_subjects = n_subjects
		self.session = session
		self.random_state = random_state

	def _normalize(self, df):
		feature_columns = [col for col in df.columns if col not in COLUMNS_TO_DROP]

		feature_min = df[feature_columns].min()
		feature_max = df[feature_columns].max()

		df[feature_columns] = (df[feature_columns] - feature_min) / (
			feature_max - feature_min + EPSILON
		)

		return df

	def split(self):
		df = pd.read_csv(self.csv_path)

		df = df[df[COLUMN_SESSION_INDEX] == self.session]

		subjects = sorted(df[COLUMN_SUBJECT].unique())[: self.n_subjects]
		df = df[df[COLUMN_SUBJECT].isin(subjects)]

		# Normalizuojame prieš dalijimą
		df = self._normalize(df)

		X = df.drop(columns=COLUMNS_TO_DROP)
		y = df[COLUMN_SUBJECT]

		X_trainval, X_test, y_trainval, y_test = train_test_split(
			X, y, test_size=SPLIT_SIZE, random_state=self.random_state, stratify=y
		)

		X_train, X_val, y_train, y_val = train_test_split(
			X_trainval,
			y_trainval,
			test_size=SPLIT_SIZE,
			random_state=self.random_state,
			stratify=y_trainval,
		)

		os.makedirs(self.output_dir, exist_ok=True)

		for name, X_split, y_split in [
			(FILENAME_TRAIN, X_train, y_train),
			(FILENAME_VALIDATION, X_val, y_val),
			(FILENAME_TEST, X_test, y_test),
		]:
			split_df = X_split.copy()
			split_df.insert(0, COLUMN_SUBJECT, y_split)
			split_df.to_csv(os.path.join(self.output_dir, f'{name}.csv'), index=False)

		print(f'Subjects: {len(subjects)}')
		print(f'Session: {self.session}')
		print(f'Total samples: {len(df)}')
		print(f'{FILENAME_TRAIN}: {len(X_train)} ({len(X_train) / len(df) * 100:.1f}%)')
		print(f'{FILENAME_VALIDATION}: {len(X_val)} ({len(X_val) / len(df) * 100:.1f}%)')
		print(f'{FILENAME_TEST}: {len(X_test)} ({len(X_test) / len(df) * 100:.1f}%)')

		print('Keystroke Data Split Completed!\n')
