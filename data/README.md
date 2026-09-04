# Data

Real participant data is **not** included in this repository. This folder documents the
expected layout so the code can be pointed at your own copy of the data via `--data_root`
(default: `./data`).

Human-subject sensor and lab-value data must not be published without confirming the
scope of IRB approval and de-identification requirements. Contact the corresponding
author for access.

## Expected layout

```
data/
├── label.xlsx           # per-subject demographics + sCr checkpoints/labels
└── preprocessed_1m/      # per-subject minute-resampled sensor stream
    └── F001.csv ... F030.csv
```

Each `F{subject_id:03}.csv` holds the subject's minute-resampled sensor stream with
columns `[HR, RMSSD, T, SpO2, Activity, Hydration]`.

Subject 25 is excluded everywhere as a known outlier (see `OUTLIER_IDS` in
`src/cimfe/data/datasets.py`).
