from abc import ABC, abstractmethod


class BaseDataSplitter(ABC):
	@abstractmethod
	def split(self):
		pass
