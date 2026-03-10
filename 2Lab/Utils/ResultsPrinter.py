from Utils.SigmoidNeuron import SigmoidNeuron


class ResultsPrinter:
	@staticmethod
	def print_summary(neuron: SigmoidNeuron, label: str, X_test, y_test):
		error, accuracy, _, _ = neuron.evaluate(X_test, y_test)

		print(f'\n{"=" * 60}')
		print(f'  {label}')
		print(f'{"=" * 60}')
		print(f'Weights: {neuron.weights}')
		print(f'Bias: {neuron.bias:.6f}')
		print(f'Epochs run: {neuron.epochs_run}')
		print(f'Training time: {neuron.training_time:.4f}s')
		print(f'Last epoch - Train error: {neuron.train_errors[-1]:.6f}')
		print(f'Last epoch - Validation error: {neuron.validation_errors[-1]:.6f}')
		print(f'Last epoch - Train accuracy: {neuron.train_accuracies[-1]:.4f}')
		print(f'Last epoch - Validation accuracy: {neuron.validation_accuracies[-1]:.4f}')
		print(f'Test error: {error:.6f}')
		print(f'Test accuracy: {accuracy:.4f}')

	@staticmethod
	def print_test_predictions(neuron, X_test, y_test):
		_, _, predictions, y_pred = neuron.evaluate(X_test, y_test)

		print(f'\n{"Sample":>8} {"Predicted":>10} {"Actual":>8} {"Raw Output":>12} {"Correct":>8}')
		print(f'{"-" * 50}')
		for i in range(len(y_test)):
			correct = 'YES' if predictions[i] == y_test[i] else 'NO'
			print(
				f'{i + 1:>8} {predictions[i]:>10} {int(y_test[i]):>8} {y_pred[i]:>12.6f} {correct:>8}'
			)
