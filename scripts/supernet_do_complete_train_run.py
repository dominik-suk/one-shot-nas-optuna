import os
import time
from datetime import datetime

import pandas as pd
import torch

from src.models.supernet import Supernet
from src.nas.supernet_trainer import train_supernet
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

    train_supernet(
        supernet=supernet,
        epochs=250,
        save_path=SUPERNET_PATH,
        do_save=True
    )

    end_time = time.time()
    log_supernet_train_minutes(minutes=(end_time - start_time) / 60.0)


if __name__ == '__main__':
    main()