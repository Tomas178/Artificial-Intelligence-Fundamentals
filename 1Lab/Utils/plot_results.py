import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def _plot_class_points(ax, class_data: NDArray[np.floating], color: str, label: str) -> None:
	"""Helper to plot scatter points for a specific class with consistent styling"""
	ax.scatter(
		class_data[:, 0],
		class_data[:, 1],
		c=color,
		marker='o',
		s=80,
		edgecolors='black',
		label=label,
		zorder=5,
	)


def plot_results(
	X: NDArray[np.floating],
	y: NDArray[np.int_],
	found_step: list[tuple[float, float, float]],
	graphics_file_path: str,
) -> None:
	"""Paint the graphic"""

	class_0 = X[y == 0]
	class_1 = X[y == 1]

	_, ax = plt.subplots(1, 1, figsize=(10, 8))

	_plot_class_points(ax, class_0, 'blue', '0 Klasė ')
	_plot_class_points(ax, class_1, 'red', '1 Klasė ')

	colors: list[str] = ['green', 'purple', 'orange']

	x_range = np.linspace(-5, 5, 300)

	for i, (w1, w2, w0) in enumerate(found_step):
		y_line = -(w1 * x_range + w0) / w2

		ax.plot(
			x_range,
			y_line,
			color=colors[i],
			linewidth=2,
			label=f'Tiesė {i + 1}: {w1:.2f}·x₁ + {w2:.2f}·x₂ + {w0:.2f} = 0',
		)

		start_x = 0.0
		start_y = -w0 / w2

		w_vector: NDArray[np.floating] = np.array([w1, w2])

		w_norm = float(np.linalg.norm(w_vector))

		w_unit: NDArray[np.floating] = w_vector / w_norm * 1.5

		ax.annotate(
			'',
			xy=(start_x + w_unit[0], start_y + w_unit[1]),
			xytext=(start_x, start_y),
			arrowprops=dict(arrowstyle='->', color=colors[i], lw=2.5),
		)

		ax.plot(
			start_x,
			start_y,
			'o',
			color=colors[i],
			markersize=6,
			zorder=6,
		)

		ax.annotate(
			f'w{i + 1}',
			xy=(start_x + w_unit[0], start_y + w_unit[1]),
			fontsize=10,
			fontweight='bold',
			color=colors[i],
			xytext=(5, 5),
			textcoords='offset points',
		)

	ax.set_xlim(-5, 5)
	ax.set_ylim(-5, 5)

	AXIS_TITLE_FONT_SIZE = 13
	ax.set_xlabel('x₁', fontsize=AXIS_TITLE_FONT_SIZE)
	ax.set_ylabel('x₂', fontsize=AXIS_TITLE_FONT_SIZE)

	ax.set_title(
		'Dirbtinio neurono klasifikavimas:\nduomenų taškai, skiriančios tiesės ir svorių vektoriai',
		fontsize=14,
	)

	ax.legend(loc='upper left', fontsize=9)
	ax.set_aspect('equal')
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig(graphics_file_path, dpi=150, bbox_inches='tight')
	plt.close()
