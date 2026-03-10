from Utils.DataCleaner import DataCleaner
from Utils.DataSplitter import DataSplitter
from Utils.ResultsPrinter import ResultsPrinter
from Utils.SigmoidNeuron import SigmoidNeuron
from Utils.Visualizer import Visualizer

DEFAULT_LEARNING_RATE = 0.5
DEFAULT_EPOCHS_COUNT = 500


def main():
	# 1. Duomenų paruošimas
	data_cleaner = DataCleaner('dataset/breast-cancer-wisconsin.data')
	features, labels = data_cleaner.clean()
	data_cleaner.save('dataset/cleaned_data.csv')

	data_splitter = DataSplitter(features, labels)
	X_train, y_train, X_val, y_val, X_test, y_test = data_splitter.split()

	n_features = X_train.shape[1]
	visualizer = Visualizer()

	# Pagrindinis mokymas (lr=DEFAULT_LEARNING_RATE, epochs=DEFAULT_EPOCHS_COUNT)
	bgd_neuron = SigmoidNeuron(
		n_features, learning_rate=DEFAULT_LEARNING_RATE, epochs=DEFAULT_EPOCHS_COUNT
	)
	bgd_neuron.train_batch(X_train, y_train, X_val, y_val)
	ResultsPrinter.print_summary(
		bgd_neuron, f'Batch GD (lr={DEFAULT_LEARNING_RATE})', X_test, y_test
	)
	ResultsPrinter.print_test_predictions(bgd_neuron, X_test, y_test)

	sgd_neuron = SigmoidNeuron(
		n_features, learning_rate=DEFAULT_LEARNING_RATE, epochs=DEFAULT_EPOCHS_COUNT
	)
	sgd_neuron.train_stochastic(X_train, y_train, X_val, y_val)
	ResultsPrinter.print_summary(
		sgd_neuron, f'Stochastic GD (lr={DEFAULT_LEARNING_RATE})', X_test, y_test
	)
	ResultsPrinter.print_test_predictions(sgd_neuron, X_test, y_test)

	# Vizualizacijos: paklaida ir tikslumas nuo epochų
	visualizer.plot_error(bgd_neuron, f'Batch GD (lr={DEFAULT_LEARNING_RATE})', 'error_bgd.png')
	visualizer.plot_accuracy(
		bgd_neuron, f'Batch GD (lr={DEFAULT_LEARNING_RATE})', 'accuracy_bgd.png'
	)
	visualizer.plot_error(
		sgd_neuron, f'Stochastic GD (lr={DEFAULT_LEARNING_RATE})', 'error_sgd.png'
	)
	visualizer.plot_accuracy(
		sgd_neuron, f'Stochastic GD (lr={DEFAULT_LEARNING_RATE})', 'accuracy_sgd.png'
	)

	# BGD vs SGD palyginimas
	visualizer.plot_bgd_vs_sgd(bgd_neuron, sgd_neuron, 'bgd_vs_sgd.png')

	# Mokymosi greičio tyrimas
	learning_rates = [0.01, 0.1, DEFAULT_LEARNING_RATE, 0.9]
	bgd_lr_results = []
	sgd_lr_results = []

	for lr in learning_rates:
		bgd = SigmoidNeuron(n_features, learning_rate=lr, epochs=DEFAULT_EPOCHS_COUNT)
		bgd.train_batch(X_train, y_train, X_val, y_val)
		bgd_test_acc = bgd.evaluate(X_test, y_test)[1]
		bgd_lr_results.append((lr, bgd, bgd_test_acc))

		sgd = SigmoidNeuron(n_features, learning_rate=lr, epochs=DEFAULT_EPOCHS_COUNT)
		sgd.train_stochastic(X_train, y_train, X_val, y_val)
		sgd_test_acc = sgd.evaluate(X_test, y_test)[1]
		sgd_lr_results.append((lr, sgd, sgd_test_acc))

		print(
			f'lr={lr}: BGD val_acc={bgd.validation_accuracies[-1]:.4f}, SGD val_acc={sgd.validation_accuracies[-1]:.4f}'
		)

	visualizer.plot_learning_rate_comparison(bgd_lr_results, 'Batch GD', 'lr_comparison_bgd.png')
	visualizer.plot_learning_rate_comparison(
		sgd_lr_results, 'Stochastic GD', 'lr_comparison_sgd.png'
	)

	# Mokymo laiko palyginimas esant vienodam epochų skaičiui
	epoch_counts = [100, 300, DEFAULT_EPOCHS_COUNT]
	bgd_times = []
	sgd_times = []

	for ep in epoch_counts:
		bgd = SigmoidNeuron(n_features, learning_rate=DEFAULT_LEARNING_RATE, epochs=ep, e_min=0.0)
		bgd.train_batch(X_train, y_train, X_val, y_val)
		bgd_times.append(bgd.training_time)

		sgd = SigmoidNeuron(n_features, learning_rate=DEFAULT_LEARNING_RATE, epochs=ep, e_min=0.0)
		sgd.train_stochastic(X_train, y_train, X_val, y_val)
		sgd_times.append(sgd.training_time)

		print(f'Epochs={ep}: BGD={bgd.training_time:.4f}s, SGD={sgd.training_time:.4f}s')

	visualizer.plot_time_comparison(bgd_times, sgd_times, epoch_counts, 'time_comparison.png')


if __name__ == '__main__':
	main()
