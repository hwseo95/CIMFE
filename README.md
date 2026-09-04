# CIMFE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22295979.svg)](https://doi.org/10.5281/zenodo.22295979)

**Supplementary code for:**

> **An All-in-One Wearable Platform Offers a Continuous Assessment and Early Diagnosis
> of Acute Kidney Injury for Outdoor Workers**
> Hoon Yi, Roxana C. Chicas, Hyunwoo Seo, Yixuan Liu, Seunghyeb Ban, Hodam Kim, Tae Woog
> Kang, Jimin Lee, Ka Ram Kim, Ernesto Ruiz, Jeff M. Sands, Chiehyeon Lim, Li Xiong,
> Vicki Hertzberg, Woon-Hong Yeo
> TODO: journal/venue, year, and a link (DOI or arXiv) once available.

This repository covers the model-side analysis of the paper: self-supervised
pretraining of the CIMFE patch-transformer encoder on wearable sensor streams (heart
rate, HRV/RMSSD, temperature, SpO2, activity, hydration), embedding extraction, and
subject-grouped RandomForest prediction of serum creatinine (sCr) change. It does
**not** include the raw-data preprocessing step (raw device exports -> per-minute
CSVs) or the manuscript's figure-generation notebooks.

Pipeline:
1. **Pretrain** `SCRSSL` self-supervised (masked reconstruction + an optional
   physiological-strain-index auxiliary task) on raw sensor windows.
2. **Extract** a per-(subject, checkpoint) embedding from the pretrained encoder.
3. **Combine** that embedding with hand-crafted sensor statistics and demographics.
4. **Predict** sCr change with a subject-grouped RandomForest (`GroupKFold`, so no
   subject's data leaks across train/test).

## Installation

```bash
git clone https://github.com/hwseo95/CIMFE.git
cd CIMFE
pip install -e .
# or: pip install -r requirements.txt
```

Requires Python >= 3.9 and PyTorch >= 2.0 (GPU recommended for pretraining).

## Data

**This repository does not include any raw or de-identified patient/participant data,
nor any trained model checkpoints.** Place your own data under `./data/` following the
layout documented in [`data/README.md`](data/README.md), or point `--data_root` /
`DATA_ROOT` at wherever it lives. See `data/README.md` for the IRB/access note.

## Usage

Pretrain the encoder:

```bash
python scripts/train.py --data_root ./data \
    --seq_len 60 --patch_len 5 --stride 5 --e_layers 3 --d_model 128 --d_ff 256 \
    --pretrain_task psi --pretrain_epochs 100
```

This saves checkpoints under `--pretrain_checkpoints` (default `./outputs/pretrain_checkpoints/`)
named after the run's hyperparameters (see `pretrain_setting` in `scripts/train.py`).

`scripts/sweep_gpu0.sh` / `scripts/sweep_gpu1.sh` run a full pretraining sweep split
across two GPUs.

Then run the embedding + statistics + RandomForest pipeline:

```bash
cd analysis
python run_pipeline.py   # or open cell-by-cell in an interactive Python session
```

Point `total_ckpt_path` / `pretrain_setting` at the top of `run_pipeline.py` to the
checkpoint produced by the pretraining run above.

## Repository layout

```
src/cimfe/          importable library: data loading, model, self-supervised pretraining loop, utilities
src/baselines/      embedding/statistics tables + subject-grouped RandomForest evaluation
scripts/            pretraining CLI entry point (train.py) and sweep scripts
analysis/           run_pipeline.py: embedding extraction -> feature fusion -> sCr prediction
data/               (empty) expected data layout, see data/README.md
```

Both `cimfe` and `baselines` are installed as top-level packages by `pip install -e .`,
so `scripts/` and `analysis/` can `import cimfe` / `import baselines` directly.

## Citation

If you use this code, please cite the software via its Zenodo DOI:

```bibtex
@software{cimfe2026,
  author  = {Seo, Hyunwoo},
  title   = {CIMFE: Wearable-Sensor-Based Serum Creatinine (sCr) Change Prediction},
  year    = {2026},
  doi     = {10.5281/zenodo.22295979},
  url     = {https://github.com/hwseo95/CIMFE}
}
```

Once the accompanying paper above is published, please also (or instead) cite it --
this section will be updated with its full journal reference and DOI.

See [`CITATION.cff`](CITATION.cff) for the canonical, machine-readable citation metadata
(also used by GitHub's "Cite this repository" button), which includes both the software
citation and a `preferred-citation` entry for the paper.

## License

MIT, see [`LICENSE`](LICENSE).
