import os
import shutil

from sklearn.model_selection import train_test_split

from Utils.consts import DATA_FOLDERS, FILENAME_VALIDATION
from Utils.DataSplitter import BaseDataSplitter
from Utils.DataSplitter.consts import RANDOM_STATE, SPLIT_SIZE


class ImageDataSplitter(BaseDataSplitter):
	def __init__(self, train_dir, output_dir, val_ratio=SPLIT_SIZE, random_state=RANDOM_STATE):
		self.train_dir = train_dir
		self.output_dir = output_dir
		self.val_ratio = val_ratio
		self.random_state = random_state

	def split(self):
		for cls in os.listdir(self.train_dir):
			cls_dir = os.path.join(self.train_dir, cls)
			if not os.path.isdir(cls_dir):
				continue

			val_cls_dir = os.path.join(self.output_dir, FILENAME_VALIDATION, cls)
			os.makedirs(val_cls_dir, exist_ok=True)

			files = os.listdir(cls_dir)
			_, val_files = train_test_split(
				files, test_size=self.val_ratio, random_state=self.random_state
			)

			for file in val_files:
				shutil.move(os.path.join(cls_dir, file), os.path.join(val_cls_dir, file))

		for split in DATA_FOLDERS:
			split_dir = os.path.join(self.output_dir, split)
			if not os.path.isdir(split_dir):
				continue
			for cls in sorted(os.listdir(split_dir)):
				count = len(os.listdir(os.path.join(split_dir, cls)))
				print(f'{split}/{cls}: {count} images')

		print('Image Data Split Completed!\n')
