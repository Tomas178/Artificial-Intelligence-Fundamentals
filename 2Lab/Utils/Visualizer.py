import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from Utils.SigmoidNeuron import SigmoidNeuron


class Visualizer:
	def __init__(self, output_dir: str = 'visualizations', colormap: str = 'viridis'):
		self.output_dir = output_dir
		self.cmap = cm.get_cmap(colormap)
		os.makedirs(output_dir, exist_ok=True)

	def _colors(self, n: int) -> list:
		return [self.cmap(i / max(n - 1, 1)) for i in range(n)]

	def _save(self, filename: str):
		filepath = os.path.join(self.output_dir, filename)
		plt.tight_layout()
		plt.savefig(filepath, dpi=150, bbox_inches='tight')
		plt.close()
		print(f'Saved: {filepath}')

	def plot_error(self, neuron: SigmoidNeuron, label: str, filename: str):
		c = self._colors(2)
		epochs = range(1, neuron.epochs_run + 1)

		plt.figure(figsize=(10, 6))
		plt.plot(epochs, neuron.train_errors, label='Training Error', color=c[0], linewidth=2)
		plt.plot(
			epochs, neuron.validation_errors, label='Validation Error', color=c[1], linewidth=2
		)
		plt.xlabel('Epoch')
		plt.ylabel('MSE')
		plt.title(f'Error vs Epochs - {label}')
		plt.legend()
		plt.grid(True, alpha=0.3)
		self._save(filename)

	def plot_accuracy(self, neuron: SigmoidNeuron, label: str, filename: str):
		c = self._colors(2)
		epochs = range(1, neuron.epochs_run + 1)

		plt.figure(figsize=(10, 6))
		plt.plot(
			epochs, neuron.train_accuracies, label='Training Accuracy', color=c[0], linewidth=2
		)
		plt.plot(
			epochs,
			neuron.validation_accuracies,
			label='Validation Accuracy',
			color=c[1],
			linewidth=2,
		)
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.title(f'Accuracy vs Epochs - {label}')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.ylim([0.5, 1.05])
		self._save(filename)

	def plot_learning_rate_comparison(self, results: list, method_label: str, filename: str):
		c = self._colors(3)
		lrs = [str(lr) for lr, _, _ in results]
		val_accs = [n.validation_accuracies[-1] for _, n, _ in results]
		test_accs = [ta for _, _, ta in results]
		val_errs = [n.validation_errors[-1] for _, n, _ in results]

		x = np.arange(len(lrs))
		width = 0.35

		_, axes = plt.subplots(1, 2, figsize=(14, 6))

		axes[0].bar(x - width / 2, val_accs, width, label='Validation Acc', color=c[0])
		axes[0].bar(x + width / 2, test_accs, width, label='Test Acc', color=c[1])
		axes[0].set_xlabel('Learning Rate')
		axes[0].set_ylabel('Accuracy')
		axes[0].set_title(f'Accuracy by Learning Rate - {method_label}')
		axes[0].set_xticks(x)
		axes[0].set_xticklabels(lrs)
		axes[0].legend()
		axes[0].set_ylim([0.8, 1.02])
		axes[0].grid(True, alpha=0.3, axis='y')

		axes[1].bar(x, val_errs, width, color=c[2])
		axes[1].set_xlabel('Learning Rate')
		axes[1].set_ylabel('Validation Error (MSE)')
		axes[1].set_title(f'Validation Error by Learning Rate - {method_label}')
		axes[1].set_xticks(x)
		axes[1].set_xticklabels(lrs)
		axes[1].grid(True, alpha=0.3, axis='y')

		self._save(filename)

	def plot_bgd_vs_sgd_error(
		self, bgd_neuron: SigmoidNeuron, sgd_neuron: SigmoidNeuron, filename='bgd_vs_sgd_error.png'
	):
		c = self._colors(4)
		epochs_bgd = range(1, bgd_neuron.epochs_run + 1)
		epochs_sgd = range(1, sgd_neuron.epochs_run + 1)

		plt.figure(figsize=(10, 6))
		plt.plot(epochs_bgd, bgd_neuron.train_errors, label='BGD Train', color=c[0], linewidth=2)
		plt.plot(
			epochs_bgd,
			bgd_neuron.validation_errors,
			label='BGD Val',
			color=c[1],
			linestyle='--',
			linewidth=2,
		)
		plt.plot(epochs_sgd, sgd_neuron.train_errors, label='SGD Train', color=c[2], linewidth=2)
		plt.plot(
			epochs_sgd,
			sgd_neuron.validation_errors,
			label='SGD Val',
			color=c[3],
			linestyle='--',
			linewidth=2,
		)
		plt.xlabel('Epoch')
		plt.ylabel('MSE')
		plt.title('Error: BGD vs SGD')
		plt.legend()
		plt.grid(True, alpha=0.3)
		self._save(filename)

	def plot_bgd_vs_sgd_accuracy(
		self,
		bgd_neuron: SigmoidNeuron,
		sgd_neuron: SigmoidNeuron,
		filename='bgd_vs_sgd_accuracy.png',
	):
		c = self._colors(4)
		epochs_bgd = range(1, bgd_neuron.epochs_run + 1)
		epochs_sgd = range(1, sgd_neuron.epochs_run + 1)

		plt.figure(figsize=(10, 6))
		plt.plot(
			epochs_bgd, bgd_neuron.train_accuracies, label='BGD Train', color=c[0], linewidth=2
		)
		plt.plot(
			epochs_bgd,
			bgd_neuron.validation_accuracies,
			label='BGD Val',
			color=c[1],
			linestyle='--',
			linewidth=2,
		)
		plt.plot(
			epochs_sgd, sgd_neuron.train_accuracies, label='SGD Train', color=c[2], linewidth=2
		)
		plt.plot(
			epochs_sgd,
			sgd_neuron.validation_accuracies,
			label='SGD Val',
			color=c[3],
			linestyle='--',
			linewidth=2,
		)
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.title('Accuracy: BGD vs SGD')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.ylim([0.5, 1.05])
		self._save(filename)

	def plot_bgd_vs_sgd_val_accuracy(
		self,
		bgd_neuron: SigmoidNeuron,
		sgd_neuron: SigmoidNeuron,
		filename='bgd_vs_sgd_val_accuracy.png',
	):
		c = self._colors(4)
		methods = ['BGD', 'SGD']
		val_accs = [bgd_neuron.validation_accuracies[-1], sgd_neuron.validation_accuracies[-1]]

		plt.figure(figsize=(10, 6))
		plt.bar(methods, val_accs, color=[c[0], c[2]])
		plt.ylabel('Validation Accuracy')
		plt.title('Final Validation Accuracy')
		plt.ylim([0.8, 1.02])
		plt.grid(True, alpha=0.3, axis='y')
		self._save(filename)

	def plot_bgd_vs_sgd_time(
		self, bgd_neuron: SigmoidNeuron, sgd_neuron: SigmoidNeuron, filename='bgd_vs_sgd_time.png'
	):
		c = self._colors(4)
		methods = ['BGD', 'SGD']
		times = [bgd_neuron.training_time, sgd_neuron.training_time]

		plt.figure(figsize=(10, 6))
		plt.bar(methods, times, color=[c[0], c[2]])
		plt.ylabel('Time (seconds)')
		plt.title('Training Time')
		plt.grid(True, alpha=0.3, axis='y')
		self._save(filename)

	def plot_time_comparison(
		self, bgd_times: list, sgd_times: list, epoch_counts: list, filename='time_comparison.png'
	):
		c = self._colors(2)
		x = np.arange(len(epoch_counts))
		width = 0.35

		plt.figure(figsize=(10, 6))
		plt.bar(x - width / 2, bgd_times, width, label='Batch GD', color=c[0])
		plt.bar(x + width / 2, sgd_times, width, label='Stochastic GD', color=c[1])
		plt.xlabel('Number of Epochs')
		plt.ylabel('Training Time (seconds)')
		plt.title('Training Time: BGD vs SGD')
		plt.xticks(x, [str(e) for e in epoch_counts])
		plt.legend()
		plt.grid(True, alpha=0.3, axis='y')
		self._save(filename)
