import os

from consts import DATASETS_DIRECTORY, KEYSTROKES_DIRECTORY, MUFFIN_VS_CHIHUAHUA_DIRECTORY
from Utils.DataReader import ImageDataReader, KeystrokeDataReader


def main():
	# Paloadiname nuotraukas
	IMAGE_DIR = os.path.join(DATASETS_DIRECTORY, MUFFIN_VS_CHIHUAHUA_DIRECTORY)

	image_data_reader = ImageDataReader(IMAGE_DIR)
	dataset_images_train, dataset_images_validation, dataset_images_test = image_data_reader.read()

	# Paloadiname keystrokes duomenis
	KEYSTROKE_DIR = os.path.join(DATASETS_DIRECTORY, KEYSTROKES_DIRECTORY)

	keystroke_data_reader = KeystrokeDataReader(KEYSTROKE_DIR)

	train_df, validation_df, test_df = keystroke_data_reader.read()


if __name__ == '__main__':
	main()
