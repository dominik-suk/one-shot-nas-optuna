from pathlib import Path

from matplotlib import pyplot as plt


def safe_save(figure: plt.Figure, destination: Path):
    new_path = get_alternative_path(destination)
    figure.savefig(new_path)


def get_alternative_path(filepath: Path) -> Path:
    if not filepath.exists():
        return filepath

    i = 1
    while True:
        new_path = filepath.parent / f"{filepath.stem}_{i}{filepath.suffix}"

        if not new_path.exists():
            return new_path

        i += 1
