import os
import time
import torch
import zipfile
import numpy as np
import pandas as pd
import urllib.request
from enum import Enum
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class Pamap2ActivityType(Enum):
    ALL = 'all'
    PROTOCOL = 'protocol'
    ADL = 'adl'


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class Pamap2Dataset(Dataset):
    URL = 'https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip'

    _PROTOCOL_ACTIVITIES = {1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24}
    _ACTIVITIES_OF_DAILY_LIVING = {1, 2, 3, 4, 12, 13}
    _ALL_ACTIVITIES = {
        1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
        6: "cycling", 7: "nordic walking", 9: "watching TV", 10: "computer work",
        11: "car driving", 12: "ascending stairs", 13: "descending stairs",
        16: "vacuum cleaning", 17: "ironing", 18: "folding laundry",
        19: "house cleaning", 20: "playing soccer", 24: "rope jumping",
    }

    SPLITS = {
        'train': ['subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat', 'subject107.dat', 'subject108.dat'],
        'validation': ['subject105.dat'],
        'test': ['subject106.dat'],
    }

    def __init__(
            self,
            root: str,
            split: str,
            activity_type: Pamap2ActivityType,
            sequence_length: int,
            mean: torch.Tensor = None,
            std: torch.Tensor = None,
    ):
        self.root = root
        self.split = split
        self.activity_type = activity_type
        self.sequence_length = sequence_length
        self.stride = sequence_length // 2
        self.mean = mean
        self.std = std

        self.valid_ids = sorted(list(self._ALL_ACTIVITIES.keys()))
        self.id_to_index = {activity_id: index for index, activity_id in enumerate(self.valid_ids)}

        self.protocol_dir = os.path.join(root, 'PAMAP2_Dataset', 'Protocol')
        self.optional_dir = os.path.join(root, 'PAMAP2_Dataset', 'Optional')
        self.processed_file = os.path.join(root, f"pamap2_{split}_{activity_type.value}.pt")

        if not os.path.exists(self.processed_file):
            self._download_and_unzip()
            self._perform_preprocessing()

        if not os.path.exists(self.processed_file):
            raise RuntimeError(f"Dataset not found or processed.")

        self.features, self.labels = torch.load(self.processed_file, weights_only=True)
        self._normalize()
        self.num_windows = (len(self.features) - self.sequence_length) // self.stride + 1

    def __len__(self):
        return max(0, self.num_windows)

    def __getitem__(self, index):
        start_index = index * self.stride
        end_index = start_index + self.sequence_length
        window_labels = self.labels[start_index:end_index]

        x = self.features[start_index:end_index]
        y = torch.mode(window_labels).values

        return x, y

    def _download_and_unzip(self):
        os.makedirs(self.root, exist_ok=True)

        if not os.path.exists(self.protocol_dir):
            outer_zip_path = os.path.join(self.root, 'pamap2.zip')
            if not os.path.exists(outer_zip_path):
                with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="PAMAP2") as t:
                    urllib.request.urlretrieve(self.URL, outer_zip_path, reporthook=t.update_to)

            with zipfile.ZipFile(outer_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)

            inner_zip_path = os.path.join(self.root, 'PAMAP2_Dataset.zip')
            if os.path.exists(inner_zip_path):
                with zipfile.ZipFile(inner_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.root)

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

                if self.activity_type == Pamap2ActivityType.PROTOCOL:
                    df = df[df[1].isin(self._PROTOCOL_ACTIVITIES)]
                elif self.activity_type == Pamap2ActivityType.ADL:
                    df = df[df[1].isin(self._ACTIVITIES_OF_DAILY_LIVING)]

                if df.empty:
                    continue

                features = df.iloc[:, 2:].values.astype(np.float32)
                activity_ids = df[1].values.astype(int)
                labels = np.array([self.id_to_index[_id] for _id in activity_ids])

                all_features.append(features)
                all_labels.append(labels)

        features_tensor = torch.tensor(np.concatenate(all_features, axis=0))
        labels_tensor = torch.tensor(np.concatenate(all_labels, axis=0))

        torch.save((features_tensor, labels_tensor), self.processed_file)

    def _normalize(self):
        if self.mean is None or self.std is None:
            self.mean = self.features.mean(dim=0, keepdim=True)
            self.std = self.features.std(dim=0, keepdim=True)
            self.std[self.std == 0] = 1e-6

        self.features = (self.features - self.mean) / self.std

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
        root_dir='./data',
        batch_size=64,
        activity_type: Pamap2ActivityType = Pamap2ActivityType.ALL,
        sequence_length=256
):
    train_dataset = Pamap2Dataset(
        root=root_dir,
        split='train',
        activity_type=activity_type,
        sequence_length=sequence_length
    )

    validation_dataset = Pamap2Dataset(
        root=root_dir,
        split='validation',
        activity_type=activity_type,
        sequence_length=sequence_length,
        mean=train_dataset.mean,
        std=train_dataset.std,
    )

    test_dataset = Pamap2Dataset(
        root=root_dir,
        split='test',
        activity_type=activity_type,
        sequence_length=sequence_length,
        mean=train_dataset.mean,
        std=train_dataset.std,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, validation_loader, test_loader


if __name__ == '__main__':
    start_time = time.time()

    get_data(
        root_dir='../../data',
        batch_size=64,
        activity_type=Pamap2ActivityType.ALL,
        sequence_length=256
    )

    get_data(
        root_dir='../../data',
        batch_size=64,
        activity_type=Pamap2ActivityType.PROTOCOL,
        sequence_length=256
    )

    get_data(
        root_dir='../../data',
        batch_size=64,
        activity_type=Pamap2ActivityType.ADL,
        sequence_length=256
    )

    end_time = time.time()
    total_seconds = end_time - start_time
    minutes, seconds = divmod(total_seconds, 60)

    print(f'Total time: {int(minutes)} minutes and {seconds:.2f} seconds')
