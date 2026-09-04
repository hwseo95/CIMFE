import os

import numpy as np
import pandas as pd
import torch

from cimfe.data.datasets import CumulativeDataset
from cimfe.utils.tools import create_patch

OUTLIER_IDS = [25]
ALL_SUBJECT_IDS = list(range(1, 31))


def _normal_ids():
    return list(set(ALL_SUBJECT_IDS) - set(OUTLIER_IDS))


def demographic_table(data_root):
    normal_ids = np.array(_normal_ids())
    raw = pd.read_excel(os.path.join(data_root, 'label.xlsx'))[:-1]
    raw = raw.iloc[normal_ids - 1]
    demo = raw[['Age', 'Height', 'Weight', 'BMI', 'Body Fat']]
    demo['subj'] = normal_ids
    demo = demo[['subj', 'Age', 'Height', 'Weight', 'BMI', 'Body Fat']]
    return demo


def deep_embedding_table(data_root, model, normalizer, how='last'):
    """Extract one embedding per (subject, checkpoint) sample from a pretrained SCRSSL encoder."""
    dataset = CumulativeDataset(data_root, scale=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    ids = []
    embeddings = []
    trues = []
    model.eval()
    normalizer.eval()
    with torch.no_grad():
        for batch_x, target, id in loader:
            batch_x = batch_x.float().to(model.device)
            normed_x = normalizer(batch_x, 'norm')
            normed_x = create_patch(normed_x, model.patch_len, model.patch_len)
            emb = model.backbone(normed_x)
            emb = emb[:, -1, :].reshape(1, -1) if how == 'last' else emb.mean(dim=1)
            embeddings.append(emb.cpu().numpy())
            ids.append(id.item())
            trues.append(target.item())

    ids, embeddings, trues = np.array(ids), np.concatenate(embeddings, axis=0), np.array(trues)
    deep_df = pd.DataFrame(embeddings, columns=[f'd{i}' for i in range(embeddings.shape[1])])
    deep_df['subj'] = ids
    deep_df['sCr change'] = trues
    deep_df = deep_df[['subj'] + [f'd{i}' for i in range(embeddings.shape[1])] + ['sCr change']]
    return deep_df


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


def load_label(data_root):
    df = pd.read_excel(os.path.join(data_root, 'label.xlsx'))
    df = df.iloc[:-1, [0, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29]]
    df.columns = ['name'] + list(df.columns)[1:]
    df['Device time'] = 0
    for i in range(1, 7):
        df[f'Device time.{i}'] = df[f'Device time.{i}'].apply(convert_total_minute)
    df = df.apply(shift_nans_right, axis=1)
    return df


def get_all_stat_features(data):
    tmp = []
    for i in range(7):
        tmp.append(data[:, i].mean())
        tmp.append(data[-1, i] - data[0, i])
        tmp.append(np.median(data[:, i]))
        tmp.append(data[:, i].max() - data[:, i].min())
        tmp.append(data[:, i].std())
    return tmp


def stat_table(data_root):
    """Hand-crafted per-checkpoint statistical features (mean/diff/median/range/std)
    over each sensor channel, plus a derived physiological strain index (PSI)."""
    df_label = load_label(data_root)
    stats = ['mean', 'diff', 'median', 'range', 'std']
    signal_cols = ['HR', 'RMSSD', 'T', 'SpO2_selected', 'Activity', 'Hydration', 'PSI']

    all_values = []
    for n in _normal_ids():
        df = pd.read_csv(os.path.join(data_root, 'preprocessed_1m', f'F{n:03}.csv'))
        df.loc[:, 'PSI'] = 5 * ((df.loc[:, 'T'] - df.loc[0, 'T']) / (39.5 - df.loc[0, 'T']) + (df.loc[:, 'HR'] - df.loc[0, 'HR']) / (180 - df.loc[0, 'HR']))
        df = df.values
        label_arr = df_label.iloc[n - 1, 1:].values
        for i in range(6):
            values = [n]
            i *= 2
            e = int(label_arr[i + 2])
            target = label_arr[i + 3] - label_arr[1]

            data = df[:e]
            values += get_all_stat_features(data)
            values.append(target)
            all_values.append(values)

            if e == -1:
                break

    columns = ["subj"] + [f'{i}_{j}' for i in signal_cols for j in stats] + ['sCr change']
    last_df = pd.DataFrame(all_values, columns=columns)
    return last_df
