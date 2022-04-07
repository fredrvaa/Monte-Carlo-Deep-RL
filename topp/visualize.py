"""
Contains code for visualizing win ratios.
"""
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpt


def plot_win_ratios(model_names: list[str],
                    win_ratios: np.ndarray,
                    cmap: str = 'rainbow',
                    title: Optional[str] = None) -> None:
    """
    Plots win ratios for an arbitrary number of models as a bar chart.

    :param model_names: A list of names for the models.
                        Length M of list must be equal to the MxM dimensions of the win_ratios np.array
    :param win_ratios: A MxM np.array of win ratios
    :param cmap: Matplotlib cmap name used when coloring bar chart.
    :param title: Title of plot
    """

    # Setup constants
    fig = plt.figure()
    n_models = len(model_names)
    cmap = plt.cm.get_cmap(cmap, n_models)
    step = 1 / n_models

    # Setup plot
    fig.suptitle(title)
    ax = plt.subplot()
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_names)
    ax.set_xlabel('Model')
    ax.set_ylabel('Win ratio')
    for i, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(cmap(i))

    legend_patches = [mpt.Patch(color=cmap(i), label=model_name) for i, model_name in enumerate(model_names)]
    plt.legend(handles=legend_patches)

    # Plot mean win percentage
    ax.plot(range(n_models), win_ratios.sum(axis=1) / (n_models - 1), color='black', marker='o', zorder=2)

    # Plot bars
    for i, (win, model_name) in enumerate(zip(win_ratios, model_names)):
        n = 0
        for j in range(n_models):
            if i == j:
                continue
            ax.bar((i - 0.5 + step) + n * step, win[j], width=step, color=cmap(j), zorder=1)
            n += 1

    plt.show(block=True)