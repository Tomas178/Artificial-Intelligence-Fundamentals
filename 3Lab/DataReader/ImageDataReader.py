import os

import torch
from consts import FILENAME_TEST, FILENAME_TRAIN, FILENAME_VALIDATION
from DataSplitter.consts import RANDOM_STATE
from torch.utils.data import Subset
from torchvision import datasets, transforms

from DataReader.BaseDataReader import BaseDataReader

IMAGE_SIZE = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUBSET_RATIO = 0.5  # Naudojame tik 50% duomenų


class ImageDataReader(BaseDataReader):
	transform = transforms.Compose(
		[
			transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
			transforms.ToTensor(),
			transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
		]
	)

	def __init__(self, base_dir: str):
		self.base_dir = base_dir

	def _reduce_dataset(self, dataset):
		total = len(dataset)
		subset_size = int(total * SUBSET_RATIO)
		generator = torch.Generator().manual_seed(RANDOM_STATE)
		indices = torch.randperm(total, generator=generator)[:subset_size].tolist()
		subset = Subset(dataset, indices)
		subset.classes = dataset.classes  # Išsaugome klasių pavadinimus
		return subset

	def read(self):
		dataset_train = datasets.ImageFolder(
			os.path.join(self.base_dir, FILENAME_TRAIN), transform=self.transform
		)
		dataset_validation = datasets.ImageFolder(
			os.path.join(self.base_dir, FILENAME_VALIDATION), transform=self.transform
		)
		dataset_test = datasets.ImageFolder(
			os.path.join(self.base_dir, FILENAME_TEST), transform=self.transform
		)

		dataset_train = self._reduce_dataset(dataset_train)
		dataset_validation = self._reduce_dataset(dataset_validation)
		dataset_test = self._reduce_dataset(dataset_test)

		print('Images Data:')
		print(f'{FILENAME_TRAIN}: {len(dataset_train)} Images.')
		print(f'{FILENAME_VALIDATION}: {len(dataset_validation)} Images.')
		print(f'{FILENAME_TEST}: {len(dataset_test)} Images.\n')

		return dataset_train, dataset_validation, dataset_test
