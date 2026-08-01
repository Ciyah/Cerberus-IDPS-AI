# Cerberus IDPS — Run Instructions

This repository contains the two scripts that make up the Cerberus pipeline:

1. `Cerberus_Genesis_Pipeline_40_Features.py` — Phase 1 & 2: dataset ingestion, feature engineering, normalisation, cleaning, class balancing, and HGBM model training.
2. `Cerberus_Phase3_DIAD2024_Validation.py` — Phase 3: zero-shot, fine-tuned, and fresh-retrain evaluation against the CIC IoT-DIAD 2024 dataset.

The scripts must be run **in this order**, since Phase 3 loads the model produced by Phase 1 & 2.

---

## 1. Requirements

- Python 3.9+
- Packages:
  ```bash
  pip install pandas numpy scikit-learn joblib
  ```

---

## 2. Expected Folder Structure

Place both scripts in the same directory, alongside the following dataset folders (all obtained from Kaggle):

```
Framework/
├── Cerberus_Genesis_Pipeline_40_Features.py
├── Cerberus_Phase3_DIAD2024_Validation.py
├── CICIoT2023/
│   └── CICIOT23/
│       └── train/
│           └── train.csv
├── IoTID20/
│   └── IoT Network Intrusion Dataset.csv
└── CIC-DIAD/
    ├── Benign/
    │   └── *.csv
    ├── DDOS/
    │   └── *.csv
    ├── DOS/
    │   └── *.csv
    └── Spoofing/
        └── *.csv
```

- `CICIoT2023` and `IoTID20` are read directly by file path in the Genesis script.
- `CIC-DIAD` is scanned recursively (`**/*.csv`) by the Phase 3 script — subfolder depth within `Benign`, `DDOS`, `DOS`, and `Spoofing` doesn't matter, but files are matched by keyword in the folder name (`benign`, `ddos`/`dos`, `spoof`/`mitm`).

---

## 3. Step 1 — Run the Genesis Pipeline (Phase 1 & 2)

```bash
python Cerberus_Genesis_Pipeline_40_Features.py
```

This will:
- Ingest and merge CICIoT2023 (Dataset A) and IoTID20 (Dataset B)
- Extract the 40-feature extreme physics matrix from each
- Apply per-source Local Baseline Normalisation (Z-score `StandardScaler`, fitted separately on each dataset before merging)
- Drop unmapped/ambiguous threat labels
- Purge genuinely toxic (conflicting-label) and duplicate rows
- Balance all three classes (Normal / MitM / DoS) via without-replacement undersampling, capped at 100,000 rows per class
- Split 80% Train / 20% Test, with the 70%/10% Train/Validation breakdown handled internally by the classifier's `early_stopping` + `validation_fraction=0.125`
- Train the `HistGradientBoostingClassifier` (`learning_rate=0.04`, `l2_regularization=0.05`, `max_iter=1200`, `max_leaf_nodes=255`)
- Print the classification report and confusion matrix
- Save the trained model to `Cerberus_HGBM_Brain.joblib`

**Output:** `Cerberus_HGBM_Brain.joblib` — this file is required by the Phase 3 script.

---

## 4. Step 2 — Run the Phase 3 Validation (Zero-Shot / Fine-Tune / Fresh Retrain)

```bash
python Cerberus_Phase3_DIAD2024_Validation.py
```

This will:
- Load `Cerberus_HGBM_Brain.joblib` (must exist from Step 1 — the script will error and exit if it's not found)
- Ingest the CIC IoT-DIAD 2024 dataset from `CIC-DIAD/`, capped at 150,000 rows per class
- Extract the same 40-feature schema, scale continuous features with a freshly fit `StandardScaler`
- Split 50% Fine-Tune / 50% Test
- Evaluate three variants:
  - **Zero-Shot** — the loaded model, completely unmodified. This is the only true zero-day result.
  - **Fine-Tuned** — the same model, warm-started and fine-tuned on the 50% fine-tune split. This is an adaptation diagnostic, not a validation result.
  - **Fresh Retrain** — a brand-new `HistGradientBoostingClassifier` trained entirely on the DIAD fine-tune split. This is an architecture-check diagnostic, not a validation result.
- Report accuracy, macro/weighted precision/recall/F1, and per-class detail for all three variants
- Measure and print per-packet inference latency (Zero-Shot, deployed model) against the sub-4ms real-time target
- Save `feature_coverage_diagnostic.csv` (per-feature sparsity check against the DIAD dataset)
- Save `Cerberus_HGBM_Brain_DIAD_FineTuned.joblib` and `Cerberus_HGBM_Brain_DIAD_Final.joblib`

**Outputs:**
- `feature_coverage_diagnostic.csv`
- `Cerberus_HGBM_Brain_DIAD_FineTuned.joblib`
- `Cerberus_HGBM_Brain_DIAD_Final.joblib`

---

## 5. Notes

- Both scripts resolve all paths relative to their own file location (`BASE_DIR = os.path.dirname(os.path.abspath(__file__))`), so they can be run from any working directory as long as the folder structure above is preserved.
- The `.joblib` model files are not included in this repository due to size — regenerate them locally by running the two scripts above in order.
- All datasets (CICIoT2023, IoTID20, CIC IoT-DIAD 2024) were sourced publicly from Kaggle under their respective open-access licences. No proprietary or live network traffic was collected or processed at any stage.
