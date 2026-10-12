class ModelMemento:
	def __init__(self, state):
		self._state = state

	def get_state(self):
		return self._state
