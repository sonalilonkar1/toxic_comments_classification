# Model Integration Guide

This guide documents the repeatable steps for adding new models (classical or neural) to the project. Follow these steps to ensure every baseline uses the shared infrastructure (TF-IDF features, pipeline, metrics, experiments, and tests) consistently. Update this file whenever new integration requirements emerge.

---
## 1. Plan the Integration
1. **Decide feature inputs:**
   - Classical models usually reuse TF-IDF word + char n-grams via `src/features/tfidf.py` (vectorizer factory + bucket helpers) while their trainers live under `src/models/`.
   - Neural models (LSTM, DistilBERT) will have their own tokenization/data pipelines (plan in `src/data/`).
2. **Identify outputs:** ensure the new model can produce per-label probabilities (or calibrated scores) so metrics/fairness/decision policies remain consistent.
3. **Determine configuration knobs:** list the hyperparameters that should be configurable via CLI flags (e.g., regularization strength, number of estimators, learning rate).

Document any special requirements in `plan.md` before coding so they align with proposal milestones.

---
## 2. Implement the Trainer
### 2.1 Classical models (TF-IDF based)
1. **Add a trainer module** (e.g., `src/models/tfidf_linear_svm.py`) and expose a trainer function such as `train_multilabel_tfidf_linear_svm`. Pattern:
   - Accept `X_train` text list, `y_train` numpy array, label names, vectorizer params, and model-specific params.
   - Import `create_tfidf_vectorizer` (and `oversample_buckets` if needed) from `src.features.tfidf`, fit the vectorizer, and transform training text.
   - Loop over labels to fit one model per label (or a shared multi-label classifier if applicable).
   - Return the fitted vectorizer and a dict of per-label estimators. Export the trainer via `__all__` so other modules can import it consistently.
2. **Calibration:**
   - If the base model lacks `predict_proba`, wrap it with `CalibratedClassifierCV` (e.g., LinearSVC).
   - Expose calibration parameters (method, CV folds) so CLI can tune them.

### 2.2 Neural / Transformer models
1. Create a new module (e.g., `src/models/lstm.py`, `src/models/transformer.py`).
2. Implement data loading/tokenization, model definition, training loop, and evaluation hooks. Reuse shared utilities (`src/utils/metrics.py`) for evaluation.
3. Provide serialization steps for model weights and any tokenizers.

---
## 3. Update TrainConfig and Pipeline
1. **Extend `TrainConfig` (src/pipeline/train.py)`** with:
   - New `model_type` option (e.g., `"random_forest"`).
   - Default hyperparameter dicts (e.g., `rf_params`).
   - Any additional configs (e.g., calibration settings, tokenization options for neural models).
2. **Modify `_train_single_fold`:** add a branch that calls the new trainer based on `config.model_type`.
3. **Artifact persistence:**
   - Ensure `_persist_artifacts` logs `model_type` and saves any extra outputs (e.g., neural checkpoints, tokenizer files).
   - Keep the file structure consistent (overall/per-label metrics, fairness slices, predictions, models directory).

---
## 4. Extend the CLI
1. Add the new `model_type` to the `--model` choices in `src/cli/train_pipeline.py`.
2. Introduce model-specific flags (e.g., `--rf-n-estimators`, `--rf-max-depth`), wiring them into the relevant config dictionaries.
3. Update help text/usage examples so users know how to invoke the new model.

---
## 5. Write Tests
1. Update `tests/test_train_pipeline.py` (or create new test modules) to include a toy-run for the new model:
   - Create a small synthetic dataset (existing fixture `toy_data` can be reused).
   - Configure the model with tiny hyperparameters (e.g., `n_estimators=10`) to keep runtime short.
   - Assert the pipeline runs end-to-end, creating metrics files and serialized models.
2. For neural models, consider integration tests that mock/simplify training (e.g., run 1 epoch on a small subset) to validate wiring without heavy compute.

---
## 6. Documentation & Runbook
1. Update `README.md` to mention the new model availability and add example commands.
2. Add a section to `EXPERIMENT_RUNBOOK.md` detailing:
   - Command template with recommended hyperparameters.
   - Parameter meanings and how they impact accuracy, calibration, fairness, or runtime.
3. If the model introduces new artifacts (e.g., attention maps, latency logs), describe them here and in the runbook.

---
## 7. Calibration & Decision Policies (Upcoming Shared Module)
When the calibration/decision-policy module is ready:
1. Ensure the new model outputs probabilities that feed into `compute_multilabel_metrics` and the upcoming policy helpers.
2. Log Brier/ECE metrics if the model uses custom calibration (or note why not).
3. Record fixed-precision/top-K thresholds for the model in `decision_policies.json` once that feature lands.

---
## 8. Checklist Before Submitting a PR
- [ ] Trainer function implemented and returns vectorizer + per-label models.
- [ ] `TrainConfig` and pipeline support the new `model_type`.
- [ ] CLI flags documented and functional.
- [ ] Tests cover success path (and error cases if applicable).
- [ ] README + runbook updated with usage instructions.
- [ ] Artifacts verified under `experiments/` (metrics JSON/CSV, fairness, models).

Following this guide ensures every new model integrates smoothly and stays consistent with the reproducible pipeline we’re building. Update this file whenever the workflow evolves (e.g., when neural models or calibration logging ship).