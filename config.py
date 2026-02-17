import os

# Folder'iai saugojimui
data_output_dir = 'duomenys'
graphics_output_dir = 'grafikai'

# Failų pavadinimai
starting_points_data_file = 'pradiniai_duomenys.csv'
starting_points_graphics_file = 'pradiniai_taskai.png'


def get_data_path(filename: str):
	return os.path.join(data_output_dir, filename)


def get_graphics_path(filename: str):
	return os.path.join(graphics_output_dir, filename)


def create_data_dir():
	os.makedirs(data_output_dir, exist_ok=True)


def create_graphics_dir():
	os.makedirs(graphics_output_dir, exist_ok=True)


def create_all_dirs():
	create_data_dir()
	create_graphics_dir()
