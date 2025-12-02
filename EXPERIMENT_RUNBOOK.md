# Experiment Runbook

This living document explains how to execute each supported model pipeline, which parameters you can tune, and how those choices influence the resulting metrics, calibration, and analysis artifacts. Update this file whenever new models or evaluation features are added so the runbook always reflects the current capabilities.

---
## Common Setup
1. Ensure the environment is active (`conda activate toxbench` or `source .venv/bin/activate`).
2. Confirm data and splits exist (`data/raw/train.csv`, `data/splits/fold*_seed*.json`). If not, run:
   ```bash
   bash scripts/00_bootstrap_project.sh --skip-torch
   ./scripts/run_python.sh -m src.cli.make_splits --folds 3
   ```
3. Use the CLI entry point for reproducible runs:
   ```bash
   ./scripts/run_python.sh -m src.cli.train_pipeline [OPTIONS]
   ```
   Outputs land under `experiments/tfidf_logreg/` in timestamped folders containing metrics, fairness slices, predictions, and serialized models.

Key shared flags:
- `--fold FOLD_NAME`: run a specific chronological fold; omit to run all folds.
- `--normalization {raw,toy,rich}`: controls preprocessing aggressiveness. Rich normalization generally improves robustness to obfuscation but may slightly dampen rare token signals.
- `--max-features N`, `--ngram-max K`: tune TF-IDF vocabulary size. Larger vocabularies can boost recall but increase runtime/memory.
- `--bucket-col COL`, `--bucket-mult tag=factor`: enable bucket-aware oversampling to lift minority behaviors (effects mainly observed in recall for bucket-aligned labels).
- `--threshold T`: probability threshold used for metrics (micro/macro F1, hamming loss). Decision-policy targets will override this in future updates.

Artifacts to monitor:
- `overall_metrics.json`: micro/macro precision/recall/F1, hamming loss, subset accuracy.
- `per_label_metrics.csv`: precision/recall/F1/support per label.
- `fairness_slices.csv`: subgroup metrics (when identity cols available).
- `test_predictions.csv`: per-row probabilities/predictions for downstream analysis.
- `models/`: serialized TF-IDF vectorizer + per-label estimators for reuse.

---
## Logistic Regression Baseline (TF-IDF + One-vs-Rest)

**Command template:**
```bash
./scripts/run_python.sh -m src.cli.train_pipeline \
  --fold fold1_seed42 \
  --model logistic \
  --normalization toy \
  --max-features 50000 \
  --model-C 1.0 \
  --model-max-iter 400 \
  --output-dir experiments/tfidf_logreg
```

**Key parameters and their effects:**
- `--model-C`: inverse regularization strength. Higher values reduce bias (potentially higher recall) but risk overfitting/noisier calibration.
- `--model-max-iter`: ensure convergence; increase if you see convergence warnings.
- Class weighting is enabled by default (`class_weight="balanced"`), which typically boosts macro F1 by improving minority-label recall.

**Metrics to monitor:**
- Micro F1 reflects aggregate performance, useful when class prevalence matters.
- Macro F1 captures balance across labels (particularly `threat`, `identity_hate`).
- Examine SHAP outputs (from notebooks) to interpret n-grams driving predictions.

---
## Linear SVM Baseline (TF-IDF + Calibrated LinearSVC)

**Command template with illustrative hyperparameters:**
```bash
./scripts/run_python.sh -m src.cli.train_pipeline \
  --fold fold1_seed42 \
  --model svm \
  --normalization rich \
  --max-features 80000 \
  --svm-C 0.75 \
  --svm-max-iter 3000 \
  --svm-class-weight balanced \
  --svm-calib-method sigmoid \
  --svm-calib-cv 5 \
  --output-dir experiments/tfidf_logreg
```

**Parameter guidance:**
- `--svm-C`: similar trade-offs as logistic C. Lower values improve generalization but may reduce recall.
- `--svm-class-weight`: set to `balanced` to mitigate label skew; leaving it unset biases toward majority classes but may slightly improve precision.
- `--svm-calib-method {sigmoid,isotonic}`: sigmoid (Platt) is faster; isotonic can yield better calibration when dev data is plentiful.
- `--svm-calib-cv`: more folds stabilize calibration at the cost of runtime.

**Result interpretation:**
- Compare calibrated probabilities (Brier/ECE in future logging) against logistic outputs; SVM often yields sharper margins but needs calibration for trustworthy thresholds.
- Inspect fairness slices to see if the margin-based model shifts subgroup gaps compared to logistic.

---
## Random Forest Baseline (TF-IDF + One-vs-Rest RF)

**Command template:**
```bash
./scripts/run_python.sh -m src.cli.train_pipeline \
  --fold fold1_seed42 \
  --model random_forest \
  --normalization toy \
  --max-features 60000 \
  --rf-n-estimators 500 \
  --rf-max-depth 30 \
  --rf-class-weight balanced \
  --rf-max-features sqrt \
  --output-dir experiments/tfidf_logreg
```

**Parameter guidance:**
- `--rf-n-estimators`: more trees generally improve stability but increase runtime; start with 300–600.
- `--rf-max-depth`: cap depth (e.g., 30–50) to avoid overfitting rare classes; `None` grows full trees.
- `--rf-max-features`: `sqrt` is a solid default; `log2` reduces variance; integers allow deterministic feature counts.
- `--rf-class-weight`: `balanced` boosts recall on minority labels; `None` may yield higher precision on majority labels.
- `--rf-min-samples-split/leaf`: raise these (e.g., 5/2) to smooth predictions when data is noisy.
- `--rf-n-jobs`: set to `-1` to parallelize across cores; use smaller values on constrained machines.

**Result interpretation:**
- RF can outperform linear models on non-linear token interactions but may be slower and memory-heavy with large vocabularies.
- Examine per-label precision/recall; trees sometimes boost `threat`/`identity_hate` recall when class weights are enabled.
- Compare fairness slices to see if tree ensembles reduce subgroup gaps relative to linear methods.

---
## Decision Policies (Upcoming Enhancements)
We plan to add automatic logging for:
- **Fixed-precision thresholds** (e.g., ≥90% precision) using dev split tuning, reporting recall and alert counts on test data.
- **Top-K review policies** (alerts per 10k comments) with precision/recall trade-offs.
- **Calibration quality** metrics (Brier score, Expected Calibration Error) stored alongside model artifacts.

Once implemented, additional CLI flags will allow specifying target precision/K values, and new files (e.g., `calibration_metrics.json`, `decision_policies.json`) will appear under each experiment folder.

---
## Future Models
Use this section to document upcoming baselines as they land:
- **Naive Bayes / XGBoost:** planned to reuse TF-IDF features with model-specific hyperparameters and calibration wrappers.
- **LSTM / DistilBERT:** will introduce tokenization configs, learning-rate schedulers, and GPU requirements; latency and calibration measurements will be logged alongside accuracy.

For each new model, add:
1. Command template(s).
2. Parameter descriptions + recommended ranges.
3. Notes on how changes impact accuracy, calibration, fairness, or interpretability metrics.

---
## Maintenance Checklist
- After adding a model or feature, verify the runbook covers: setup command, parameter meanings, metric interpretation, and artifact expectations.
- Keep the document synced with CLI flags (`src/cli/train_pipeline.py`) and pipeline outputs (`src/pipeline/train.py`).
- Encourage contributors to append changelog entries or notes when they discover impactful parameter settings.
