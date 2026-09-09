import optuna

from src.data.pamap2_loader import Pamap2ActivityType
from src.models.supernet import Supernet
from src.nas.supernet_trainer import train_supernet
from src.paths import SUPERNET_DIR
from src.utils.yaml_io import load_pamap2_search_space, load_pamap2_fixed_arch_config


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    search_space = load_pamap2_search_space()
    fixed_architecture_config = load_pamap2_fixed_arch_config()

    supernet = Supernet(search_space)

    train_supernet(
        supernet=supernet,
        search_space=search_space,
        fixed_architecture_config=fixed_architecture_config,
        epochs=100,
        activity_type=Pamap2ActivityType.ADL,
        save_path=SUPERNET_DIR
    )


if __name__ == '__main__':
    main()