import math
import os
from pathlib import Path

import optuna
from matplotlib import pyplot as plt

from src.paths import BAR_CHARTS_DIR
from src.utils.optuna_io import get_spos_study, get_baseline_study
from src.utils.plot_io import safe_save


def get_top_10_percent_params(study: optuna.Study):
    trials = study.trials
    trials.sort(key=lambda x: x.value, reverse=True)
    n_top_10_percent = int(len(trials) * 0.1)
    top_trials = trials[:n_top_10_percent]

    return [trial.params for trial in top_trials]


def get_all_possible_values(all_trials: list[optuna.trial.FrozenTrial]) -> dict:
    all_params = [trial.params for trial in all_trials]
    possible_values = {}
    for params in all_params:
        for param_name, param_value in params.items():
            if param_name not in possible_values:
                possible_values[param_name] = set()
            possible_values[param_name].add(str(param_value))

    return possible_values


def initialize_frequency_dict(possible_values: dict) -> dict:
    frequencies = {}
    for param_name, values in possible_values.items():
        try:
            sorted_values = sorted(list(values), key=float)
        except ValueError:
            sorted_values = sorted(list(values), key=str)

        frequencies[param_name] = {val: 0 for val in sorted_values}

    return frequencies

def count_category_frequencies(prams_to_count: list[dict], all_trials: list[optuna.trial.FrozenTrial]) -> dict:
    possible_values = get_all_possible_values(all_trials)
    param_counts = initialize_frequency_dict(possible_values)

    for params in prams_to_count:
        for param_name, param_value in params.items():
            param_counts[param_name][str(param_value)] += 1

    return param_counts


def filter_useless_params(param_counts: dict) -> dict:
    return {
        name: counts for name, counts in param_counts.items()
        if parameter_appears_at_least_n_times(counts, n=10) and
           parameter_has_more_than_one_value(counts)
    }


def parameter_has_more_than_one_value(counts: dict) -> bool:
    return len(counts) > 1


def parameter_appears_at_least_n_times(counts: dict, n : int) -> bool:
    return sum(counts.values()) > n


def format_param_name(param_name: str):
    parts = param_name.split('/')
    cleaned_parts = [part.replace('_', ' ').title() for part in parts]

    if len(cleaned_parts) >= 2:
        return f"{cleaned_parts[0]}\n" + " > ".join(cleaned_parts[1:])
    return cleaned_parts[0]


def format_title_to_filepath(title: str, suffix: str = '.png') -> Path:
    return BAR_CHARTS_DIR / f"{title.replace(' ', '_')}{suffix}"


def plot_bar_chart(title: str, study: optuna.Study, do_save: bool = False):
    top_10_percent_params = get_top_10_percent_params(study)
    param_frequencies = count_category_frequencies(top_10_percent_params, study.trials)
    param_frequencies = filter_useless_params(param_frequencies)

    cols = 3
    num_params = len(param_frequencies)
    rows = math.ceil(num_params / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()
    best_params = study.best_trial.params

    for i, (param_name, counts) in enumerate(param_frequencies.items()):
        labels = list(counts.keys())
        values = list(counts.values())

        colors = ["skyblue"] * len(labels)
        if param_name in best_params:
            best_value = str(best_params[param_name])

            if best_value in labels:
                colors[labels.index(best_value)] = "orange"

        ax = axes[i]
        bars = ax.bar(labels, values, color=colors, edgecolor='black')
        ax.bar_label(bars, padding=3, fontsize=9, color='black')
        max_value = max(values)
        ax.set_ylim(0, max_value + (max_value * 0.2))
        ax.set_title(format_param_name(param_name))
        ax.set_ylabel('Frequency')
        ax.tick_params(axis='x', rotation=45)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=0.9, wspace=0.3, bottom= 0.05, top=0.9)

    if do_save:
        figure_path = format_title_to_filepath(title, suffix='.jpg')
        os.makedirs(figure_path.parent, exist_ok=True)
        safe_save(fig, figure_path)

    plt.show()


def spos_plot_bar_chart(title: str, random_search: bool = False, do_save: bool = False):
    study = get_spos_study(random_search)
    plot_bar_chart(title, study, do_save)


def baseline_plot_bar_chart(title: str, do_save: bool = False):
    study = get_baseline_study()
    plot_bar_chart(title, study, do_save)


def main():
    spos_plot_bar_chart(
        title="SPOS NAS Category Frequency Analysis",
        do_save=True,
    )
    spos_plot_bar_chart(
        title="SPOS NAS Random Search Category Frequency Analysis",
        do_save=True,
    )
    baseline_plot_bar_chart(
        title="Baseline NAS Category Frequency Analysis",
        do_save=True,
    )


if __name__ == "__main__":
    main()
