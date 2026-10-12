from collections.abc import Iterable, Iterator

from Utils.best_epoch_summary import best_epoch_summary


class OutcomeCollection(Iterable):
	def __init__(self, outcomes=()):
		self._outcomes = list(outcomes)

	def extend(self, outcomes):
		self._outcomes.extend(outcomes)

	def __iter__(self):
		return iter(self._outcomes)

	def ranked(self):
		return RankedIterator(self._outcomes)

	def by_study(self):
		return StudyIterator(self._outcomes)


class RankedIterator(Iterator):
	def __init__(self, outcomes):
		self._outcomes = sorted(outcomes, key=self._score, reverse=True)
		self._position = 0

	@staticmethod
	def _score(outcome):
		summary = best_epoch_summary(outcome['history'])
		return summary['val_acc'], -summary['val_loss']

	def __next__(self):
		if self._position >= len(self._outcomes):
			raise StopIteration
		outcome = self._outcomes[self._position]
		self._position += 1
		return outcome


class StudyIterator(Iterator):
	def __init__(self, outcomes):
		self._groups = {}
		for outcome in outcomes:
			self._groups.setdefault(outcome['study'], []).append(outcome)
		self._studies = list(self._groups)
		self._position = 0

	def __next__(self):
		if self._position >= len(self._studies):
			raise StopIteration
		study = self._studies[self._position]
		self._position += 1
		return study, OutcomeCollection(self._groups[study])
