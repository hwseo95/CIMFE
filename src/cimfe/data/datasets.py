import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# Subject 25 was excluded as a labeled outlier during data collection.
OUTLIER_IDS = [25]
ALL_SUBJECT_IDS = list(range(1, 31))


def _normal_ids():
    return list(set(ALL_SUBJECT_IDS) - set(OUTLIER_IDS))


def get_n_windows(data_root, window_size=60, stride=1):
    n_windows = 0
    for n in _normal_ids():
        df = pd.read_csv(os.path.join(data_root, 'preprocessed_1m', f'F{n:03}.csv')).values
        n_windows += (len(df) - window_size) // stride + 1
    return n_windows


class PTDataset(Dataset):
    """Pretraining dataset: sliding windows over the raw sensor stream."""

    def __init__(self, data_root, window_size=60, stride=1, scale=True, indices=None, flag='train'):
        self.data_root = data_root
        self.window_size = window_size
        self.stride = stride
        self.patch_len = stride
        self.scale = scale
        self.flag = flag
        self.indices = indices
        self._read_data()
        self._get_flag_data()

    def _read_data(self):
        x_data = []
        psi_data = []
        subject_ids = []
        for n in _normal_ids():
            df = pd.read_csv(os.path.join(self.data_root, 'preprocessed_1m', f'F{n:03}.csv'))
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(df)
            psi = (df.iloc[:, 2].values - df.iloc[0, 2]) / (39.5 - df.iloc[0, 2]) + (df.iloc[:, 0].values - df.iloc[0, 0]) / (180 - df.iloc[0, 0])
            l = len(df)
            n_data = (max(l - self.window_size, 0) // self.stride) + 1

            for i in range(n_data):
                s, e = i * self.patch_len, i * self.patch_len + self.window_size
                batch_x = x_scaled[s:e, :]
                batch_psi = psi[s:e]
                x_data.append(batch_x)
                psi_data.append(batch_psi)
            subject_ids += [n] * n_data

        self.x_data = torch.tensor(np.array(x_data)).float()
        self.psi_data = torch.tensor(np.array(psi_data)).float()
        self.subject_ids = torch.tensor(subject_ids).long()
        print(self.flag, len(self.x_data))

    def _get_flag_data(self):
        if self.indices is not None:
            self.x_data = self.x_data[self.indices]

    def __getitem__(self, index):
        seq_x = self.x_data[index]
        psi = self.psi_data[index]
        subj = self.subject_ids[index]

        return seq_x, psi, subj

    def __len__(self):
        return len(self.x_data)


def convert_total_minute(t):
    try:
        return t.hour * 60 + t.minute
    except AttributeError:
        if isinstance(t, str):
            return -1
        elif np.isnan(t):
            return t


def shift_nans_right(row):
    """Move all NaNs in a row to the right so valid entries stay left-aligned."""
    non_nans = row.dropna().values
    nans = np.full((len(row) - len(non_nans)), np.nan)
    return pd.Series(np.concatenate([non_nans, nans]))


class CumulativeDataset(Dataset):
    """Cumulative sensor windows up to each labeled sCr checkpoint, one sample per
    (subject, checkpoint). Used to extract per-sample embeddings from a pretrained
    SCRSSL encoder (see baselines.tables.deep_embedding_table)."""

    def __init__(self, data_root, scale=True):
        self.data_root = data_root
        self.scale = scale
        self._read_data()

    def _load_label(self):
        df = pd.read_excel(os.path.join(self.data_root, 'label.xlsx'))
        df = df.iloc[:, [0, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29]]
        df.columns = ['name'] + list(df.columns)[1:]
        df['Device time'] = 0
        for i in range(1, 7):
            df[f'Device time.{i}'] = df[f'Device time.{i}'].apply(convert_total_minute)
        df = df.apply(shift_nans_right, axis=1)
        return df

    def _read_data(self):
        df_label = self._load_label()
        self.x_data = []
        self.y_data = []
        self.ids = []

        for n in _normal_ids():
            df = pd.read_csv(os.path.join(self.data_root, 'preprocessed_1m', f'F{n:03}.csv')).values
            if self.scale:
                scaler = StandardScaler()
                df = scaler.fit_transform(df)
            label_arr = df_label.iloc[n - 1, 1:].values
            baseline = label_arr[1]
            for i in range(6):
                i *= 2
                e = int(label_arr[i + 2])
                target = label_arr[i + 3] - baseline

                data = df[:e]
                self.x_data.append(torch.tensor(data).float())
                self.y_data.append(target)
                self.ids.append(n)

                if e == -1:
                    break

        self.y_data = torch.tensor(self.y_data)
        self.ids = torch.tensor(self.ids)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.ids[index]

    def __len__(self):
        return len(self.y_data)


def data_provider(args, flag):
    """Pretraining data loaders. `flag` is 'train' or 'val'.

    `Exp_SCRSSL.pretrain()` calls this once per flag, so the split must be
    reproducible across those two separate calls -- otherwise the "val" call
    draws a different random train_indices internally and its complement is
    not actually disjoint from what "train" trained on. A seeded RandomState
    (keyed off n_windows + args.seed, not global numpy random state) makes
    both calls agree on the same split.
    """
    n_windows = get_n_windows(args.data_root, args.seq_len, 1)
    rng = np.random.RandomState(args.seed)
    train_indices = rng.choice(n_windows, int(n_windows * args.train_ratio), replace=False)
    val_indices = list(set(range(n_windows)) - set(train_indices))

    if flag == 'train':
        dataset = PTDataset(args.data_root, window_size=args.seq_len, stride=1, scale=True, flag='train', indices=train_indices)
    elif flag == 'val':
        dataset = PTDataset(args.data_root, window_size=args.seq_len, stride=1, scale=True, flag='val', indices=val_indices)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    return dataset, dataloader
