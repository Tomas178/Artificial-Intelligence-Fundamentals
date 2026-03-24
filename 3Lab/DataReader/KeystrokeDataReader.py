import os

import pandas as pd
from consts import FILENAME_TEST, FILENAME_TRAIN, FILENAME_VALIDATION

from DataReader import BaseDataReader


class KeystrokeDataReader(BaseDataReader):
	def __init__(self, base_dir: str):
		self.base_dir = base_dir

	def read(self):
		train_df = pd.read_csv(os.path.join(self.base_dir, f'{FILENAME_TRAIN}.csv'))
		validation_df = pd.read_csv(os.path.join(self.base_dir, f'{FILENAME_VALIDATION}.csv'))
		test_df = pd.read_csv(os.path.join(self.base_dir, f'{FILENAME_TEST}.csv'))

		print('Keystrokes Data:')
		print(f'{FILENAME_TRAIN}: {len(train_df)} Objects.')
		print(f'{FILENAME_VALIDATION}: {len(validation_df)} Objects.')
		print(f'{FILENAME_TEST}: {len(test_df)} Objects.\n')

		return train_df, validation_df, test_df
