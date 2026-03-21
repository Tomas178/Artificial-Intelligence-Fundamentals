import os
import shutil

from sklearn.model_selection import train_test_split


class DataSplitter:
	def __init__(self, train_dir, output_dir, val_ratio=0.2, random_state=42):
		self.train_dir = train_dir
		self.output_dir = output_dir
		self.val_ratio = val_ratio
		self.random_state = random_state

	def split(self):
		for cls in os.listdir(self.train_dir):
			cls_dir = os.path.join(self.train_dir, cls)
			if not os.path.isdir(cls_dir):
				continue

			val_cls_dir = os.path.join(self.output_dir, 'validation', cls)
			os.makedirs(val_cls_dir, exist_ok=True)

			files = os.listdir(cls_dir)
			_, val_files = train_test_split(
				files, test_size=self.val_ratio, random_state=self.random_state
			)

			for f in val_files:
				shutil.move(os.path.join(cls_dir, f), os.path.join(val_cls_dir, f))

		print('Split complete!')
		for split in ['train', 'validation', 'test']:
			split_dir = os.path.join(self.output_dir, split)
			if not os.path.isdir(split_dir):
				continue
			for cls in sorted(os.listdir(split_dir)):
				count = len(os.listdir(os.path.join(split_dir, cls)))
				print(f'  {split}/{cls}: {count} images')
