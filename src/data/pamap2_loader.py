import os
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.data.pamap2_labels import Pamap2ActivityType
from src.paths import PAMAP2_DIR, PAMAP2_DATA_LOADERS_DIR


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class Pamap2Dataset(Dataset):
    URL = 'https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip'
    SPLITS = {
        'train': ['subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat', 'subject107.dat', 'subject108.dat'],
        'validation': ['subject105.dat', 'subject106.dat'],
        'test': ['subject105.dat', 'subject106.dat'],
    }

    def __init__(
            self,
            split: str,
            activity_type: Pamap2ActivityType,
            sequence_length: int,
    ):
        self.split = split
        self.activity_type = activity_type
        self.sequence_length = sequence_length
        self.stride = sequence_length // 2

        self.valid_ids = sorted(activity_type.valid_ids)
        self.protocol_dir = PAMAP2_DIR / 'PAMAP2_Dataset' / 'Protocol'
        self.optional_dir = PAMAP2_DIR / 'PAMAP2_Dataset' / 'Optional'
        self.processed_file = PAMAP2_DATA_LOADERS_DIR / activity_type.value.upper() / f'PAMAP2_{activity_type.value.upper()}_{split.capitalize()}_Data.pth'

        if not os.path.exists(self.processed_file):
            os.makedirs(self.processed_file.parent, exist_ok=True)
            self._download_and_unzip()
            self._perform_preprocessing()

        if not os.path.exists(self.processed_file):
            raise RuntimeError(f"Dataset not found or processed.")

        self.features, self.labels = torch.load(self.processed_file, weights_only=True)
        self.num_windows = (len(self.features) - self.sequence_length) // self.stride + 1

    def __len__(self):
        return max(0, self.num_windows)

    def __getitem__(self, index):
        start_index = index * self.stride
        end_index = start_index + self.sequence_length
        window_labels = self.labels[start_index:end_index]

        x = self.features[start_index:end_index].clone()
        x = self._normalize(x)
        x = x.transpose(0, 1)

        y = window_labels[-1]                   # -> Activity on last time step
        # y = torch.mode(window_labels).values  # -> Most common activity in sequence

        return x, y

    def _download_and_unzip(self):
        if not os.path.exists(self.protocol_dir):
            outer_zip_path = os.path.join(PAMAP2_DIR, 'pamap2.zip')
            if not os.path.exists(self.protocol_dir.parent):
                with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="PAMAP2") as t:
                    urllib.request.urlretrieve(self.URL, outer_zip_path, reporthook=t.update_to)

            with zipfile.ZipFile(outer_zip_path, 'r') as zip_ref:
                zip_ref.extractall(PAMAP2_DIR)

            inner_zip_path = os.path.join(PAMAP2_DIR, 'PAMAP2_Dataset.zip')
            if os.path.exists(inner_zip_path):
                with zipfile.ZipFile(inner_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(PAMAP2_DIR)

            os.remove(outer_zip_path)
            os.remove(inner_zip_path)


    def _perform_preprocessing(self):
        subjects = self.SPLITS[self.split]
        all_features = []
        all_labels = []

        data_directories = [self.protocol_dir, self.optional_dir]

        for subject_file in subjects:
            for directory in data_directories:
                file_path = os.path.join(directory, subject_file)
                if not os.path.exists(file_path):
                    continue

                df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='c')

                df = self._interpolate_missing_heart_rate(df)
                df = self._fill_missing_data(df)
                df = self._filter_out_invalid_ids(df)

                if df.empty:
                    continue

                features = df.iloc[:, 2:].values.astype(np.float32)
                activity_ids = df[1].values.astype(int)
                id_to_index = self.activity_type.valid_id_index_pair
                labels = np.array([id_to_index[_id] for _id in activity_ids])

                all_features.append(features)
                all_labels.append(labels)

        features_tensor = torch.tensor(np.concatenate(all_features, axis=0))
        labels_tensor = torch.tensor(np.concatenate(all_labels, axis=0))

        torch.save((features_tensor, labels_tensor), self.processed_file)

    @staticmethod
    def _normalize(x: np.ndarray):
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std[std == 0] = 1e-6
        return (x - mean) / std

    @staticmethod
    def _interpolate_missing_heart_rate(df: pd.DataFrame) -> pd.DataFrame:
        df[2] = df[2].interpolate(method="linear")
        return df

    @staticmethod
    def _fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.ffill().bfill()
        return df

    def _filter_out_invalid_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df[1].isin(self.valid_ids)]


def get_data(
        batch_size=64,
        activity_type: Pamap2ActivityType = Pamap2ActivityType.ALL,
        sequence_length=256
):
    """
    :return: Train Dataloader, Validation Dataloader, Testing Dataloader
    """
    train_dataset = Pamap2Dataset(
        split='train',
        activity_type=activity_type,
        sequence_length=sequence_length
    )

    validation_dataset = Pamap2Dataset(
        split='validation',
        activity_type=activity_type,
        sequence_length=sequence_length,
    )

    test_dataset = Pamap2Dataset(
        split='test',
        activity_type=activity_type,
        sequence_length=sequence_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, validation_loader, test_loader


if __name__ == '__main__':
    start_time = time.time()

    get_data(
        batch_size=64,
        activity_type=Pamap2ActivityType.ALL,
        sequence_length=256
    )

    get_data(
        batch_size=64,
        activity_type=Pamap2ActivityType.PROTOCOL,
        sequence_length=256
    )

    get_data(
        batch_size=64,
        activity_type=Pamap2ActivityType.ADL,
        sequence_length=256
    )

    end_time = time.time()
    total_seconds = end_time - start_time
    minutes, seconds = divmod(total_seconds, 60)

    print(f'Total time: {int(minutes)} minutes and {seconds:.2f} seconds')
