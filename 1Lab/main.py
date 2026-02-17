import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Enums.Activation_function import ActivationFunction
from numpy.typing import NDArray
from tabulate import tabulate
from Utils.find_weights import find_weights
from Utils.plot_results import plot_results
from Utils.print_verification import print_verification

sys.path.append(str(Path(__file__).parent.parent))
from Config import config

config.set_lab_dir(str(Path(__file__).parent))


def main():
	# Gauname pradinius duomenis
	df = pd.read_csv(config.get_shared_data_path(config.starting_points_data_file))

	X: NDArray[np.floating] = df[['x1', 'x2']].values
	y: NDArray[np.int_] = df['klase'].values.astype(int)

	print('Duomenys sėkmingai nuskaityti')
	print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=True))

	# Svorių ir poslinkio paieška – SLENKSTINĖ aktyvacijos funkcija
	print('Svorių paieška (slenkstinė aktyvacijos f-ja)')

	np.random.seed(123)
	found_step: list[tuple[float, float, float]] = find_weights(X, y, ActivationFunction.STEP)

	# Svorių ir poslinkio paieška – SIGMOIDINĖ aktyvacijos funkcija
	print('Svorių paieška (sigmoidinė aktyvacijos f-ja)')

	np.random.seed(456)
	found_sigmoid: list[tuple[float, float, float]] = find_weights(X, y, ActivationFunction.SIGMOID)

	# 5 & 6. Vizualizacija
	config.create_graphics_dir()
	graphics_file_path: str = config.get_graphics_path('neuronas.png')

	plot_results(X, y, found_step, graphics_file_path)

	print(f'Grafikas išsaugotas: {graphics_file_path}')

	# Patikrinimas
	print_verification(X, y, found_step, found_sigmoid)


if __name__ == '__main__':
	main()
