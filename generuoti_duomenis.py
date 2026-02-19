from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Config import config

config.set_lab_dir(str(Path(__file__).parent))

config.create_all_dirs()

np.random.seed(42)

STANDARD_DEVIATION = 0.8

class_0 = np.random.randn(15, 2) * STANDARD_DEVIATION + np.array([-2, -2])

class_1 = np.random.randn(15, 2) * STANDARD_DEVIATION + np.array([2, 2])

X = np.vstack([class_0, class_1])
y = np.array([0] * 15 + [1] * 15)

data_full_path = config.get_data_path(config.starting_points_data_file)
data = np.column_stack([X, y])
np.savetxt(
	data_full_path,
	data,
	delimiter=',',
	header='x1,x2,klase',
	comments='',
	fmt=['%.4f', '%.4f', '%d'],
)
print(f'Duomenys išsaugoti: {data_full_path}')

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
	class_0[:, 0],
	class_0[:, 1],
	c='blue',
	marker='o',
	s=80,
	edgecolors='black',
	label='0 Klasė',
)
ax.scatter(
	class_1[:, 0],
	class_1[:, 1],
	c='red',
	marker='o',
	s=80,
	edgecolors='black',
	label='1 Klasė',
)

min_val = np.floor(X.min())
max_val = np.ceil(X.max())

ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)

ticks = np.arange(min_val, max_val + 1, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)

ax.set_xlabel('x₁', fontsize=13)
ax.set_ylabel('x₂', fontsize=13)
ax.set_title('Sugeneruoti duomenų taškai', fontsize=14)
ax.legend(loc='upper left')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

graphics_full_path = config.get_graphics_path(config.starting_points_graphics_file)
plt.tight_layout()
plt.savefig(
	graphics_full_path,
	dpi=105,
	bbox_inches='tight',
)
plt.close()
print(f'Grafikas išsaugotas: {graphics_full_path}')
