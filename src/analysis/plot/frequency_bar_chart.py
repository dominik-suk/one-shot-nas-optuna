import re
from pathlib import Path

import matplotlib as mpl
import optuna
from matplotlib import pyplot as plt

from src.paths import BAR_CHARTS_DIR
from src.utils.optuna_io import get_baseline_study, get_spos_study
from src.utils.plot_io import safe_save


def get_top_10_percent_params(study: optuna.Study):
    trials = study.trials
    trials = [trial for trial in trials if trial.value is not None]
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


def count_category_frequencies(
    prams_to_count: list[dict], all_trials: list[optuna.trial.FrozenTrial]
) -> dict:
    possible_values = get_all_possible_values(all_trials)
    param_counts = initialize_frequency_dict(possible_values)

    for params in prams_to_count:
        for param_name, param_value in params.items():
            param_counts[param_name][str(param_value)] += 1

    return param_counts


def filter_useless_params(param_counts: dict, n: int) -> dict:
    return {
        name: counts
        for name, counts in param_counts.items()
        if parameter_appears_at_least_n_times(counts, n)
        and parameter_has_more_than_one_value(counts)
    }


def parameter_has_more_than_one_value(counts: dict) -> bool:
    return len(counts) > 1


def parameter_appears_at_least_n_times(counts: dict, n: int) -> bool:
    return sum(counts.values()) > n


def format_param_name(param_name: str) -> str:
    parts = param_name.split("/")
    cleaned_parts = [part.replace("_", " ").title() for part in parts]

    if len(cleaned_parts) >= 2:
        return f"{cleaned_parts[0]}\n" + " $\\rightarrow$ ".join(cleaned_parts[1:])
    return cleaned_parts[0]


def sanitize_param_filename(param_name: str) -> str:
    clean = param_name.strip("/").replace("/", "_")
    return re.sub(r"[^\w\-.]", "_", clean)


def get_next_run_dir(base_dir: Path, folder_prefix: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_indices = []

    pattern = re.compile(rf"^{re.escape(folder_prefix)}_(\d+)$")
    for entry in base_dir.iterdir():
        if entry.is_dir():
            match = pattern.match(entry.name)
            if match:
                existing_indices.append(int(match.group(1)))

    next_idx = max(existing_indices, default=-1) + 1
    run_dir = base_dir / f"{folder_prefix}_{next_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def plot_and_save_single_bar_chart(
    prefix: str,
    param_name: str,
    counts: dict,
    best_value: str | None,
    save_dir: Path | None = None,
    show_plot: bool = False,
):
    labels = list(counts.keys())
    values = list(counts.values())

    colors = ["skyblue"] * len(labels)
    if best_value is not None and best_value in labels:
        colors[labels.index(best_value)] = "orange"

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    ax.bar_label(bars, padding=3, fontsize=9, color="black")

    max_value = max(values) if values else 0
    ax.set_ylim(0, max_value + (max_value * 0.2) if max_value > 0 else 1)
    ax.set_title(format_param_name(param_name), fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    if save_dir is not None:
        filename = f"{prefix}_{sanitize_param_filename(param_name)}.png"
        filepath = save_dir / filename
        safe_save(fig, filepath)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_bar_charts(prefix: str, study: optuna.Study, do_save: bool = False):
    top_10_percent_params = get_top_10_percent_params(study)
    param_frequencies = count_category_frequencies(top_10_percent_params, study.trials)
    param_frequencies = filter_useless_params(
        param_frequencies, n=max(1, len(top_10_percent_params) // 100)
    )

    if not param_frequencies:
        print(f"No interesting parameter frequencies found for {prefix}")
        return

    save_dir = None
    if do_save:
        save_dir = get_next_run_dir(BAR_CHARTS_DIR, f"{prefix}_nas")
        print(f"Saving {len(param_frequencies)} plots to: {save_dir}")

    best_params = study.best_trial.params

    for param_name, counts in param_frequencies.items():
        best_value = str(best_params[param_name]) if param_name in best_params else None
        plot_and_save_single_bar_chart(
            prefix=prefix,
            param_name=param_name,
            counts=counts,
            best_value=best_value,
            save_dir=save_dir,
            show_plot=not do_save,
        )


def plot_spos_bar_chart(random_search: bool = False, do_save: bool = False):
    study = get_spos_study(random_search)
    prefix = "spos_random" if random_search else "spos"
    plot_bar_charts(prefix=prefix, study=study, do_save=do_save)


def plot_baseline_bar_chart(do_save: bool = False):
    study = get_baseline_study()
    plot_bar_charts(prefix="baseline", study=study, do_save=do_save)


if __name__ == "__main__":
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    })

    plot_spos_bar_chart(random_search=False, do_save=True)
    plot_spos_bar_chart(random_search=True, do_save=True)
    plot_baseline_bar_chart(do_save=True)