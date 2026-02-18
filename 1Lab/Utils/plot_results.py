import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_results(
	X: NDArray[np.floating],
	y: NDArray[np.int_],
	found_step: list[tuple[float, float, float]],
	graphics_file_path: str,
) -> None:
	class_0: NDArray[np.floating] = X[y == 0]
	class_1: NDArray[np.floating] = X[y == 1]

	_, ax = plt.subplots(1, 1, figsize=(10, 8))

	ax.scatter(
		class_0[:, 0],
		class_0[:, 1],
		c='blue',
		marker='o',
		s=80,
		edgecolors='black',
		label='0 Klasė',
		zorder=5,
	)
	ax.scatter(
		class_1[:, 0],
		class_1[:, 1],
		c='red',
		marker='o',
		s=80,
		edgecolors='black',
		label='1 Klasė',
		zorder=5,
	)

	colors: list[str] = ['green', 'purple', 'orange']
	x_range = np.linspace(-5, 5, 300)

	for i, (w1, w2, b) in enumerate(found_step):
		if abs(w2) > 1e-9:
			y_line = -(w1 * x_range + b) / w2
			ax.plot(
				x_range,
				y_line,
				color=colors[i],
				linewidth=2,
				label=f'Tiesė {i + 1}: {w1:.2f}·x₁ + {w2:.2f}·x₂ + {b:.2f} = 0',
			)
		else:
			x_vert: float = -b / w1
			ax.axvline(
				x=x_vert,
				color=colors[i],
				linewidth=2,
				label=f'Tiesė {i + 1}: x₁ = {x_vert:.2f}',
			)

		if abs(w2) > 1e-9:
			start_x = 0.0
			start_y = -b / w2
		else:
			start_x = -b / w1
			start_y = 0.0

		w_vec: NDArray[np.floating] = np.array([w1, w2])
		w_norm = float(np.linalg.norm(w_vec))
		w_unit: NDArray[np.floating] = w_vec / w_norm * 1.5

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
	ax.set_xlabel('x₁', fontsize=13)
	ax.set_ylabel('x₂', fontsize=13)
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
