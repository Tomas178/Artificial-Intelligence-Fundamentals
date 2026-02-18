import matplotlib.pyplot as plt
import numpy as np

from Config import config

# Sukūriame bazinius folder'ius
config.create_all_dirs()

# Dvi klasės po 15 taškų
np.random.seed(42)

STANDARD_DEVIATION = 0.8

# Klasė 0: taškai aplink centrą (-2, -2)
class_0 = np.random.randn(15, 2) * STANDARD_DEVIATION + np.array([-2, -2])

# Klasė 1: taškai aplink centrą (2, 2)
class_1 = np.random.randn(15, 2) * STANDARD_DEVIATION + np.array([2, 2])

# Sujungiame duomenis
X = np.vstack([class_0, class_1])
y = np.array([0] * 15 + [1] * 15)

# Išsaugome į CSV failą
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

# Atvaizduojame taškus Dekarto koordinačių sistemoje
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
ax.set_xlabel('x₁', fontsize=13)
ax.set_ylabel('x₂', fontsize=13)
ax.set_title('Sugeneruoti duomenų taškai', fontsize=14)
ax.legend()
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
