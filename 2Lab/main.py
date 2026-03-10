from Utils.DataCleaner import DataCleaner
from Utils.DataSplitter import DataSplitter
from Utils.ResultsPrinter import ResultsPrinter
from Utils.SigmoidNeuron import SigmoidNeuron

LEARNING_RATE = 0.5
EPOCHS = 500


def main():
	data_cleaner = DataCleaner('dataset/breast-cancer-wisconsin.data')
	features, labels = data_cleaner.clean()
	data_cleaner.save('dataset/cleaned_data.csv')

	data_splitter = DataSplitter(features, labels)
	X_train, y_train, X_validation, y_validation, X_test, y_test = data_splitter.split()

	n_features = X_train.shape[1]

	# Paketinis gradientinis nusileidimas
	bgd_neuron = SigmoidNeuron(n_features, learning_rate=LEARNING_RATE, epochs=EPOCHS)
	bgd_neuron.train_batch(X_train, y_train, X_validation, y_validation)
	ResultsPrinter.print_summary(
		bgd_neuron, f'Batch GD (lr={LEARNING_RATE}, epochs={EPOCHS})', X_test, y_test
	)
	ResultsPrinter.print_test_predictions(bgd_neuron, X_test, y_test)

	# Stochastinis gradientinis nusileidimas
	sgd_neuron = SigmoidNeuron(n_features, learning_rate=LEARNING_RATE, epochs=EPOCHS)
	sgd_neuron.train_stochastic(X_train, y_train, X_validation, y_validation)
	ResultsPrinter.print_summary(
		sgd_neuron, f'Stochastic GD (lr={LEARNING_RATE}, epochs={EPOCHS})', X_test, y_test
	)
	ResultsPrinter.print_test_predictions(sgd_neuron, X_test, y_test)


if __name__ == '__main__':
	main()
