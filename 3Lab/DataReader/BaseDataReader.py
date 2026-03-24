from abc import ABC, abstractmethod


class BaseDataReader(ABC):
	@abstractmethod
	def read(self):
		pass
