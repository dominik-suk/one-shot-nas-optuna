from pathlib import Path

import optuna
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick


from src.utils.optuna_io import get_spos_study
from src.utils.plot_io import safe_save
from src.paths import SCATTER_PLOTS_DIR

def extract_generation_data(study: optuna.Study, population_size: int = 50) -> dict[int, list[float]]:
    generations = {}

    for trial in study.trials:
        generation = (trial.number // population_size) + 1
        if generation not in generations:
            generations[generation] = []
        generations[generation].append(trial.value)

    return generations


def get_axes(generations: dict[int, list[float]], k: int = 10) -> tuple[list[int], list[float]]:
    x_plot = []
    y_plot = []

    for generation, values in generations.items():
        sorted_acc_values = sorted(values, reverse=True)
        top_k_values = sorted_acc_values[:k]
        x_plot.extend([generation] * len(top_k_values))
        y_plot.extend(top_k_values)

    return x_plot, y_plot


def plot_spos_nsga_versus_random(do_save: bool = True):
    nsga_generations = extract_generation_data(get_spos_study())
    random_generations = extract_generation_data(get_spos_study(random_search=True))

    scatter_plot(
        axes_1=get_axes(nsga_generations),
        label_1="NSGA-II",
        axes_2=get_axes(random_generations),
        label_2="Random",
        save_destination=SCATTER_PLOTS_DIR / 'SPOS_NSGA_vs_Random.png' if do_save else None
    )


def scatter_plot(axes_1: tuple, axes_2: tuple, label_1: str, label_2: str, save_destination: Path | None = None):
    x_1, y_1 = axes_1
    x_2, y_2 = axes_2

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(x_2, y_2, color='springgreen', marker='*', label=label_2, alpha=0.9)
    ax.scatter(x_1, y_1, color='orange', marker='o', label=label_1, alpha=0.8)

    ax.axhline(y=max(y_1), color='darkorange', linestyle='--', linewidth=1.2, alpha=0.9, label=f"{label_1} max ({max(y_1):.2f} \\%)")
    ax.axhline(y=max(y_2), color='seagreen', linestyle='--', linewidth=1.2, alpha=0.9, label=f"{label_2} max ({max(y_2):.2f} \\%)")


    ax.set_xlabel('Evolution iters', fontsize=12)
    ax.set_ylabel('Validation accuracy', fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    [spine.set_edgecolor('black') for spine in ax.spines.values()]
    ax.legend(loc='lower right', edgecolor='black')

    min_y = min(min(y_1), min(y_2))
    max_y = max(max(y_1), max(y_2))
    max_length = max(len(label_1), len(label_2))
    print(f"{label_1.ljust(max_length)} acc: min={min(y_1):.2f} %, max={max(y_1):.2f} %")
    print(f"{label_2.ljust(max_length)} acc: min={min(y_2):.2f} %, max={max(y_2):.2f} %")
    ax.set_ylim(min_y - 1, max_y + 1)

    plt.tight_layout()
    plt.show()

    if save_destination:
        safe_save(
            figure=fig,
            destination=save_destination
        )
