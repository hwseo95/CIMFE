#%%
# sCr-change prediction pipeline (run cell-by-cell):
#   1. load a pretrained SCRSSL encoder checkpoint (produced by `scripts/train.py`)
#   2. extract one embedding per (subject, checkpoint) sample
#   3. combine the embedding with hand-crafted sensor statistics + demographics
#   4. predict sCr change with a subject-grouped RandomForest

import pandas as pd
import warnings

from cimfe.models.scrssl import SCRSSL
from cimfe.models.layers.revin import RevIN
from cimfe.utils.tools import transfer_weights

from baselines.evaluation import evaluate_model_unseen, select_features_unseen
from baselines.tables import deep_embedding_table, stat_table, demographic_table

warnings.filterwarnings('ignore')
DATA_ROOT = '../data'
device = 'cuda:0'
model_type = 'rf'  # 'rf' or 'gb'

#%%
# Point this at a checkpoint saved under `--pretrain_checkpoints` by
# `scripts/train.py` (setting name is built from the run's hyperparameters,
# see the `pretrain_setting` string in scripts/train.py).
pretrain_setting = 'psi_dm128_df256_nh4_el3_pl5_preFalse_layernorm_ep100_plr0.005_alp0.5'
seed = 2025
total_ckpt_path = f'../outputs/pretrain_checkpoints/{pretrain_setting}/{seed}/ckpt_best.pth'


def get_pretrain_param(setting):
    task, dm, df, nh, el, pl, pre_norm, norm, ep, plr, alp = setting.split('_')
    d_model = int(dm[2:])
    d_ff = int(df[2:])
    n_heads = int(nh[2:])
    e_layers = int(el[2:])
    patch_len = int(pl[2:])
    pre_norm = eval(pre_norm[3:])
    return patch_len, d_model, d_ff, n_heads, e_layers, pre_norm, norm

#%%

patch_len, d_model, d_ff, n_heads, e_layers, pre_norm, norm = get_pretrain_param(pretrain_setting)

encoder = SCRSSL(c_in=6, patch_len=patch_len, num_patch=60 // patch_len, n_layers=e_layers,
                  d_model=d_model, d_ff=d_ff, n_heads=n_heads, pre_norm=pre_norm, norm=norm)
encoder = transfer_weights(total_ckpt_path, encoder, device=device).to(device)
normalizer = RevIN(6, affine=False).to(device)

# 1) deep embeddings from the pretrained encoder
deep_df = deep_embedding_table(DATA_ROOT, encoder, normalizer)
scores = evaluate_model_unseen(deep_df.iloc[:, :-1].values, deep_df['sCr change'].values, model=model_type, seed=seed)

# 2) hand-crafted sensor statistics, reduced to the subset that helps subject-grouped CV
stat_df = stat_table(DATA_ROOT)
stat_feats = select_features_unseen(stat_df, model=model_type, seed=seed, step=1)
stat_df = stat_df[['subj'] + stat_feats + ['sCr change']]
scores = evaluate_model_unseen(stat_df.iloc[:, :-1].values, stat_df['sCr change'].values, model=model_type, seed=seed)

#%%
def concat_demo_table(df):
    demo_df = demographic_table(DATA_ROOT)
    return pd.merge(demo_df, df, on='subj')


def concat_table(df1, df2):
    """Merge two per-subject/-checkpoint feature tables that share `subj` (id) and
    `sCr change` (target) columns, keeping df1's target column and both feature sets."""
    df1_, df2_ = df1.copy(), df2.copy()
    df2_feat_cols = list(df2.columns)[1:-1]
    df1_feat_cols = list(df1.columns)[1:-1]
    ordered_cols = ['subj'] + df1_feat_cols + df2_feat_cols + ['sCr change']
    df1_[df2_feat_cols] = df2[df2_feat_cols]
    df1_ = df1_[ordered_cols]
    return df1_

#%%
# 3) combine embeddings + selected sensor statistics + demographics, then predict sCr change
df = concat_table(deep_df, stat_df)
df = concat_demo_table(df)
X = df.iloc[:, :-1].values
y = df['sCr change'].values

scores = evaluate_model_unseen(X, y, model=model_type, seed=seed)
