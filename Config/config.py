import os

# Folder'iai saugojimui
data_output_dir = 'duomenys'
graphics_output_dir = 'grafikai'

# Failų pavadinimai
starting_points_data_file = 'pradiniai_duomenys.csv'
starting_points_graphics_file = 'pradiniai_taskai.png'

_lab_dir: str | None = None


def set_lab_dir(lab_dir: str) -> None:
	global _lab_dir
	_lab_dir = lab_dir


def _get_base() -> str:
	if _lab_dir is None:
		raise RuntimeError('Call config.set_lab_dir() first')
	return _lab_dir


def get_shared_data_path(filename: str) -> str:
	"""For data files at the project root level."""
	return os.path.join(os.path.dirname(_get_base()), data_output_dir, filename)


def get_data_path(filename: str) -> str:
	return os.path.join(_get_base(), data_output_dir, filename)


def get_graphics_path(filename: str) -> str:
	return os.path.join(_get_base(), graphics_output_dir, filename)


def create_data_dir() -> None:
	os.makedirs(os.path.join(_get_base(), data_output_dir), exist_ok=True)


def create_graphics_dir() -> None:
	os.makedirs(os.path.join(_get_base(), graphics_output_dir), exist_ok=True)


def create_all_dirs() -> None:
	create_data_dir()
	create_graphics_dir()
