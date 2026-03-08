from Utils.DataCleaner import DataCleaner
from Utils.DataSplitter import DataSplitter


def main():
	data_cleaner = DataCleaner('dataset/breast-cancer-wisconsin.data')
	features, labels = data_cleaner.clean()
	data_cleaner.save('dataset/cleaned_data.csv')

	data_splitter = DataSplitter(features, labels)
	X_train, y_train, X_validation, y_validation, X_test, y_test = data_splitter.split()


if __name__ == '__main__':
	main()
