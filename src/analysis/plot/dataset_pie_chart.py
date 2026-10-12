from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.pamap2_labels import Pamap2ActivityType, PAMAP2_ACTIVITIES
from src.data.pamap2_loader import Pamap2Dataset
from src.paths import PIE_CHARTS_DIR
from src.utils.plot_io import safe_save


def get_all_window_labels(activity_type: Pamap2ActivityType, sequence_length: int = 256):
    splits = ['train', 'validation', 'test']
    all_y = []

    for split in splits:
        dataset = Pamap2Dataset(
            split=split,
            activity_type=activity_type,
            sequence_length=sequence_length,
        )
        labels = dataset.labels
        stride = dataset.stride
        num_windows = dataset.num_windows

        windows = labels[:stride * (num_windows - 1) + sequence_length].unfold(0, sequence_length, stride)
        window_modes = torch.mode(windows, dim=1).values
        all_y.append(window_modes)

    return np.concatenate(all_y)


def plot_pie_chart(
        activity_type: Pamap2ActivityType,
        sequence_length: int = 256,
):
    window_labels = get_all_window_labels(activity_type, sequence_length)
    counts = Counter(window_labels)
    total_samples = len(window_labels)
    protocol_activity_names = activity_type.labels

    color_map = plt.get_cmap("tab20")
    activity_colormap = {
        name: color_map(i) for i, name in enumerate(PAMAP2_ACTIVITIES.values())
    }

    colors = []
    labels = []
    sizes = []
    for index, name in enumerate(protocol_activity_names):
        count = counts.get(index, 0)
        labels.append(name)
        sizes.append(count)
        colors.append(activity_colormap[name])

    percentages = [100.0 * count / total_samples for count in sizes]
    legend_labels = [f"{name} ({percentage:.2f} %)" for name, percentage in zip(labels, percentages)]

    if activity_type == Pamap2ActivityType.ALL:
        fig_size = (12, 7)
    else:
        fig_size = (10, 5)

    fig, ax = plt.subplots(figsize=fig_size)
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2.0),
    )

    ax.legend(
        wedges,
        legend_labels,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=False,
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

    safe_save(fig, PIE_CHARTS_DIR / "pamap2_dataset_pie_chart.png")


if __name__ == "__main__":
    # pprint.pprint(list(colormaps))
    plot_pie_chart(Pamap2ActivityType.PROTOCOL)
    plot_pie_chart(Pamap2ActivityType.ALL)
    plot_pie_chart(Pamap2ActivityType.ADL)