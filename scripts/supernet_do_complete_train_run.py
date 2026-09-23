import os
import time
from datetime import datetime

import pandas as pd
import torch

from src.data.pamap2_labels import get_activity_type_from_search_space
from src.data.pamap2_loader import load_pamap2_dataset
from src.logging.supernet_logger import SupernetLogger
from src.models.supernet import Supernet
from src.training.supernet_trainer import SupernetTrainer
from src.paths import SUPERNET_PATH, SUPERNET_TRAIN_TIME_LOGS
from src.utils.yaml_io import load_pamap2_search_space


def log_supernet_train_minutes(minutes: float) -> None:
    os.makedirs(SUPERNET_PATH.parent, exist_ok=True)
    file_exists = os.path.isfile(SUPERNET_TRAIN_TIME_LOGS)
    pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%m.%d.%Y - %H:%M"),
        "GPU": torch.cuda.get_device_name(),
        "Training Minutes": minutes
    }]).to_csv(SUPERNET_TRAIN_TIME_LOGS, mode="a", index=False, header=not file_exists)


def main():
    start_time = time.time()

    search_space = load_pamap2_search_space()
    supernet = Supernet(search_space)

    dataset = load_pamap2_dataset(
        activity_type=get_activity_type_from_search_space(search_space),
        sequence_length=search_space["input"][1],
    )

    supernet_trainer = SupernetTrainer(
        supernet=supernet,
        epochs=250,
        dataset=dataset,
        device='cuda',
        logger=SupernetLogger(),
        n_val_architectures=5
    )

    supernet_trainer.run()
    supernet_trainer.save_weights(path=SUPERNET_PATH)

    end_time = time.time()
    log_supernet_train_minutes(minutes=(end_time - start_time) / 60.0)


if __name__ == '__main__':
    main()