from Utils.DataSplitter import DataSplitter


def main():
	data_splitter = DataSplitter(
		train_dir='./datasets/muffin-vs-chihuahua/train',
		output_dir='./datasets/muffin-vs-chihuahua',
	)

	data_splitter.split()


if __name__ == '__main__':
	main()
