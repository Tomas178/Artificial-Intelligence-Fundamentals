import os

import pandas as pd
from consts import DATASETS_DIRECTORY, KEYSTROKES_DIRECTORY, MUFFIN_VS_CHIHUAHUA_DIRECTORY
from Utils.DataReader import ImageDataReader


def main():
	# Paloadiname nuotraukas
	IMAGE_DIR = os.path.join(DATASETS_DIRECTORY, MUFFIN_VS_CHIHUAHUA_DIRECTORY)

	image_data_reader = ImageDataReader(IMAGE_DIR)
	dataset_images_train, dataset_images_validation, dataset_images_test = image_data_reader.read()

	# Paloadiname keystrokes duomenis
	# keystrokes_dir = os.path.join(DATASETS_DIRECTORY, KEYSTROKES_DIRECTORY)

	# train_df = pd.read_csv(os.path.join(keystrokes_dir, f'{FILENAME_TRAIN}.csv'))
	# validation_df = pd.read_csv(os.path.join(keystrokes_dir, f'{FILENAME_VALIDATION}.csv'))
	# test_df = pd.read_csv(os.path.join(keystrokes_dir, f'{FILENAME_TEST}.csv'))

	# print('Keystrokes Data:')
	# print(f'{FILENAME_TRAIN}: {len(train_df)}.')
	# print(f'{FILENAME_VALIDATION}: {len(validation_df)}.')
	# print(f'{FILENAME_TEST}: {len(test_df)}.\n')


if __name__ == '__main__':
	main()
