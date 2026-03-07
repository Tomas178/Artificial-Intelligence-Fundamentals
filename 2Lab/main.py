from Utils.DataCleaner import DataCleaner


def main():
	data_cleaner = DataCleaner('dataset/breast-cancer-wisconsin.data')
	data_cleaner.clean()
	data_cleaner.save('dataset/cleaned_data.csv')


if __name__ == '__main__':
	main()
