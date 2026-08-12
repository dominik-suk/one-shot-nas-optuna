import argparse
import optuna
import os
import yaml

from src.data import pamap2_loader
from src.data.pamap2_loader import Pamap2ActivityType
from src.models.supernet import Supernet
from src.nas.supernet_trainer import train_supernet


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/PAMAP2_search_space.yaml')
    parser.add_argument('--fixed',  '-f', type=str, default='configs/PAMAP2_fixed_architecture.yaml')
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--save',   '-s', type=str, default='models/supernet')
    args = parser.parse_args()

    search_space_path = str(os.path.join(project_root, args.config))
    fixed_architecture_path = str(os.path.join(project_root, args.fixed))
    save_path = str(os.path.join(project_root, args.save))

    with open(search_space_path, 'r') as f:
        search_space = yaml.safe_load(f)

    with open(fixed_architecture_path, 'r') as f:
        validation_search_space = yaml.safe_load(f)

    supernet = Supernet(search_space)

    train_loader, validation_loader, _ = pamap2_loader.get_data(
        root_dir=os.path.join(project_root, 'data'),
        activity_type=Pamap2ActivityType.PROTOCOL
    )
    print(f"Len Training: {len(train_loader)}")
    print(f"Len Validation: {len(validation_loader)}")

    train_supernet(
        supernet,
        train_loader,
        validation_loader,
        search_space,
        validation_search_space,
        epochs=args.epochs,
        save_dir=save_path
    )


if __name__ == '__main__':
    main()