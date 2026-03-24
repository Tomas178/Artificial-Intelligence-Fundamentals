import os

from consts import DATASETS_DIRECTORY, KEYSTROKES_DIRECTORY, MUFFIN_VS_CHIHUAHUA_DIRECTORY
from DataSplitter import ImageDataSplitter, KeystrokeDataSplitter


def main():
	image_data_splitter = ImageDataSplitter(
		train_dir=os.path.join(DATASETS_DIRECTORY, MUFFIN_VS_CHIHUAHUA_DIRECTORY, 'train'),
		output_dir=os.path.join(DATASETS_DIRECTORY, MUFFIN_VS_CHIHUAHUA_DIRECTORY),
	)

	image_data_splitter.split()

	keystroke_data_splitter = KeystrokeDataSplitter(
		csv_path=os.path.join(DATASETS_DIRECTORY, 'keystrokes.csv'),
		output_dir=os.path.join(DATASETS_DIRECTORY, KEYSTROKES_DIRECTORY),
	)

	keystroke_data_splitter.split()


if __name__ == '__main__':
	main()
