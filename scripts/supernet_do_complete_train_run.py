from src.models.supernet import Supernet
from src.nas.supernet_trainer import train_supernet
from src.paths import SUPERNET_PATH
from src.utils.yaml_io import load_pamap2_search_space, load_pamap2_fixed_arch_config


def main():
    search_space = load_pamap2_search_space()
    supernet = Supernet(search_space)

    train_supernet(
        supernet=supernet,
        epochs=250,
        save_path=SUPERNET_PATH
    )


if __name__ == '__main__':
    main()