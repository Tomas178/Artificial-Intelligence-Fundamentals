import torch
from torch.utils.data import Dataset

COLUMN_SUBJECT = 'subject'


class KeystrokeDataset(Dataset):
	def __init__(self, dataframe, label_map=None):
		self.X = torch.tensor(
			dataframe.drop(columns=[COLUMN_SUBJECT]).values, dtype=torch.float32
		).unsqueeze(1)

		subjects = dataframe[COLUMN_SUBJECT]

		if label_map is None:
			unique_subjects = sorted(subjects.unique())
			self.label_map = {subject: index for index, subject in enumerate(unique_subjects)}
		else:
			self.label_map = label_map

		self.y = torch.tensor([self.label_map[subject] for subject in subjects], dtype=torch.long)
		self.classes = list(self.label_map.keys())

	def __len__(self):
		return len(self.y)

	def __getitem__(self, index):
		return self.X[index], self.y[index]
