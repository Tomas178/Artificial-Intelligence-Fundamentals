import os

from torchvision import datasets, transforms

from Utils.consts import FILENAME_TEST, FILENAME_TRAIN, FILENAME_VALIDATION
from Utils.DataReader.BaseDataReader import BaseDataReader


class ImageDataReader(BaseDataReader):
	transform = transforms.Compose(
		[
			transforms.Resize((128, 128)),
			transforms.ToTensor(),
		]
	)

	def __init__(self, base_dir: str):
		self.base_dir = base_dir

	def read(self):
		dataset_images_train = datasets.ImageFolder(
			os.path.join(self.base_dir, FILENAME_TRAIN), transform=self.transform
		)
		dataset_images_validation = datasets.ImageFolder(
			os.path.join(self.base_dir, FILENAME_VALIDATION), transform=self.transform
		)
		dataset_images_test = datasets.ImageFolder(
			os.path.join(self.base_dir, FILENAME_TEST), transform=self.transform
		)

		print('Images Data:')
		print(f'{FILENAME_TRAIN}: {len(dataset_images_train)} Images.')
		print(f'{FILENAME_VALIDATION}: {len(dataset_images_validation)} Images.')
		print(f'{FILENAME_TEST}: {len(dataset_images_test)} Images.\n')

		return dataset_images_train, dataset_images_validation, dataset_images_test
