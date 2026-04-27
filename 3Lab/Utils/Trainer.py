import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from Enums import LossFunctions, Optimizer
from sklearn.metrics import confusion_matrix

OPTIMIZER_CLASSES = {
	Optimizer.ADAM: torch.optim.Adam,
	Optimizer.SGD: torch.optim.SGD,
	Optimizer.RMSPROP: torch.optim.RMSprop,
	Optimizer.ADAMW: torch.optim.AdamW,
}

LOSS_FUNCTIONS = {
	LossFunctions.CROSS_ENTROPY: nn.CrossEntropyLoss,
	LossFunctions.NLL: nn.NLLLoss,
}


class Trainer:
	def __init__(
		self,
		model,
		train_loader,
		val_loader,
		test_loader,
		optimizer=Optimizer.ADAM,
		learning_rate=0.001,
		loss_fn=LossFunctions.CROSS_ENTROPY,
		device=None,
	):
		self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = model.to(self.device)
		self.loss_fn = LOSS_FUNCTIONS[loss_fn]()

		optimizer_class = OPTIMIZER_CLASSES[optimizer]
		self.optimizer = optimizer_class(model.parameters(), lr=learning_rate)

		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader

		self.history = {
			'train_loss': [],
			'train_acc': [],
			'val_loss': [],
			'val_acc': [],
		}

	def train(self, epochs=20):
		for epoch in range(epochs):
			train_loss, train_accuracy = self._run_epoch(self.train_loader, training=True)
			val_loss, val_accuracy = self._run_epoch(self.val_loader, training=False)

			self.history['train_loss'].append(train_loss)
			self.history['train_acc'].append(train_accuracy)
			self.history['val_loss'].append(val_loss)
			self.history['val_acc'].append(val_accuracy)

			print(
				f'Epoch {epoch + 1}/{epochs} - '
				f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} - '
				f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}'
			)

		return self.history

	def _run_epoch(self, data_loader, training):
		if training:
			self.model.train()
		else:
			self.model.eval()

		running_loss = 0
		correct_predictions = 0
		total_samples = 0

		with torch.set_grad_enabled(training):
			for inputs, labels in data_loader:
				inputs = inputs.to(self.device)
				labels = labels.to(self.device)

				outputs = self.model(inputs)
				loss = self.loss_fn(outputs, labels)

				if training:
					self.optimizer.zero_grad()
					loss.backward()
					self.optimizer.step()

				running_loss += loss.item() * inputs.size(0)
				_, predicted = outputs.max(1)
				correct_predictions += predicted.eq(labels).sum().item()
				total_samples += labels.size(0)

		average_loss = running_loss / total_samples
		accuracy = correct_predictions / total_samples
		return average_loss, accuracy

	def evaluate(self):
		test_loss, test_accuracy = self._run_epoch(self.test_loader, training=False)
		print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
		return test_loss, test_accuracy

	def _collect_predictions(self, max_samples=None):
		self.model.eval()
		prediction_chunks = []
		label_chunks = []
		total = 0

		with torch.no_grad():
			for inputs, labels in self.test_loader:
				inputs = inputs.to(self.device)
				outputs = self.model(inputs)
				_, predicted = outputs.max(1)
				prediction_chunks.append(predicted.cpu())
				label_chunks.append(labels)
				total += labels.size(0)
				if max_samples is not None and total >= max_samples:
					break

		predictions = torch.cat(prediction_chunks).numpy()
		true_labels = torch.cat(label_chunks).numpy()
		if max_samples is not None:
			predictions = predictions[:max_samples]
			true_labels = true_labels[:max_samples]
		return predictions, true_labels

	def get_confusion_matrix(self):
		predictions, true_labels = self._collect_predictions()
		return confusion_matrix(true_labels, predictions)

	def get_sample_predictions(self, num_samples=30, class_names=None):
		predictions, true_labels = self._collect_predictions(max_samples=num_samples)
		samples = []
		for predicted, true in zip(predictions, true_labels):
			true_label = class_names[int(true)] if class_names else int(true)
			predicted_label = class_names[int(predicted)] if class_names else int(predicted)
			samples.append({'true': true_label, 'predicted': predicted_label})
		return samples

	def plot_history(self, title='Training History', save_path=None):
		fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(14, 5))

		loss_ax.plot(self.history['train_loss'], label='Train')
		loss_ax.plot(self.history['val_loss'], label='Validation')
		loss_ax.set_title(f'{title} - Loss')
		loss_ax.set_xlabel('Epoch')
		loss_ax.set_ylabel('Loss')
		loss_ax.legend()

		accuracy_ax.plot(self.history['train_acc'], label='Train')
		accuracy_ax.plot(self.history['val_acc'], label='Validation')
		accuracy_ax.set_title(f'{title} - Accuracy')
		accuracy_ax.set_xlabel('Epoch')
		accuracy_ax.set_ylabel('Accuracy')
		accuracy_ax.legend()

		plt.tight_layout()
		if save_path:
			plt.savefig(save_path)
		plt.close()

	@staticmethod
	def plot_confusion_matrix(
		confusion_mat, class_names=None, title='Confusion Matrix', save_path=None
	):
		plt.figure(figsize=(10, 8))
		sns.heatmap(
			confusion_mat,
			annot=True,
			fmt='d',
			cmap='Blues',
			xticklabels=class_names,
			yticklabels=class_names,
		)
		plt.title(title)
		plt.xlabel('Predicted')
		plt.ylabel('True')
		plt.tight_layout()
		if save_path:
			plt.savefig(save_path)
		plt.close()

	@staticmethod
	def plot_comparison(results, title, save_path=None):
		fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(14, 5))

		for experiment_name, history in results.items():
			loss_ax.plot(history['val_loss'], label=experiment_name)
			accuracy_ax.plot(history['val_acc'], label=experiment_name)

		loss_ax.set_title(f'{title} - Validation Loss')
		loss_ax.set_xlabel('Epoch')
		loss_ax.set_ylabel('Loss')
		loss_ax.legend()

		accuracy_ax.set_title(f'{title} - Validation Accuracy')
		accuracy_ax.set_xlabel('Epoch')
		accuracy_ax.set_ylabel('Accuracy')
		accuracy_ax.legend()

		plt.tight_layout()
		if save_path:
			plt.savefig(save_path)
		plt.close()
