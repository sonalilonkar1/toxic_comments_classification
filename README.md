# Toxic Comment Classification - Complete ML System

A comprehensive, production-ready toxic comment classification system implementing multiple machine learning approaches with extensive evaluation, interpretability, and deployment capabilities.

## 🎯 Key Features

- **5 Model Architectures**: Logistic Regression, SVM, Random Forest, XGBoost, and BERT
- **Production-Ready Demo**: Interactive Streamlit web application with real-time predictions
- **REST API**: Flask-based API for programmatic access
- **Comprehensive Analysis**: 25+ automated visualizations and detailed performance reports
- **Hyperparameter Optimization**: Extensive tuning across all models with cross-validation
- **Interpretability**: Feature importance analysis and model explanations
- **Automated Setup**: One-click environment bootstrap and data pipeline

## 📊 Model Performance

| Model | Macro PR-AUC | Micro F1 | Macro F1 | Precision@1000 |
|-------|--------------|----------|----------|----------------|
| **BERT** | **0.722** | **0.731** | **0.466** | **0.997** |
| Linear SVM | 0.665 | 0.689 | 0.396 | 0.995 |
| Logistic Regression | 0.659 | 0.680 | 0.421 | 0.974 |
| XGBoost | 0.652 | 0.683 | 0.377 | 0.990 |
| Random Forest | 0.602 | 0.591 | 0.352 | 0.986 |

## 🚀 Quick Start

### 1. Bootstrap Everything (Recommended)

```bash
# One-command setup: environment, dependencies, PyTorch, and data
bash scripts/00_bootstrap_project.sh
```

### 2. Launch Interactive Demo

```bash
# Start the Streamlit web application
streamlit run demo.py
```

### 3. Run Model Training

```bash
# Train all models with hyperparameter tuning
./scripts/run_python.sh -m src.cli.train_pipeline --output-dir experiments/train
```

## 📋 Prerequisites

- **Python**: 3.8+ with Conda/Miniforge or virtualenv support
- **System Tools**: `git`, `curl`/`wget`, Kaggle CLI (`pip install kaggle`)
- **Kaggle Access**: API token in `~/.kaggle/kaggle.json` or project root
- **GPU (Optional)**: CUDA-compatible GPU for BERT training (CPU works but slower)

## 🏗️ Project Structure

```
toxic_comments_classification/
├── src/                          # Source code
│   ├── cli/                      # Command-line interfaces
│   ├── data/                     # Data processing and loading
│   ├── features/                 # Feature engineering
│   ├── models/                   # Model implementations
│   ├── pipeline/                 # Training pipelines
│   └── utils/                    # Utilities and helpers
├── scripts/                      # Automation scripts
│   ├── 00_bootstrap_project.sh   # One-click setup
│   ├── demo.py                   # Streamlit web app
│   ├── app.py                    # Flask REST API
│   └── create_*_plots.py         # Analysis visualization
├── experiments/                  # Experimental results
│   ├── train/                    # Trained models
│   ├── hyperparameter_tuning/    # Tuning results
│   └── bucket_augmentation/      # Data augmentation
├── reports/                      # Analysis and reports
│   ├── analysis/                 # Model comparisons
│   ├── error_analysis/           # Error patterns
│   ├── ensemble/                 # Ensemble analysis
│   ├── extended_evaluation/      # Robustness testing
│   └── figures/                  # Performance plots
├── data/                         # Datasets and splits
├── artifacts/                    # Preprocessed data
├── configs/                      # Configuration files
└── notebooks/                    # Jupyter notebooks
```

## 🎮 Interactive Demo

The Streamlit demo provides:

- **Real-time Classification**: Paste comments and see toxicity predictions
- **Multi-Model Support**: Switch between LR, SVM, RF, and BERT models
- **Interactive Thresholds**: Adjust classification thresholds
- **Top-K Ranking**: Browse highest-scoring comments from test set
- **Explainability**: Feature importance and perturbation-based explanations
- **Performance Metrics**: PR-AUC scores and confidence distributions

```bash
streamlit run demo.py
```

## 🔧 Model Training

### Train Individual Models

```bash
# Logistic Regression
./scripts/run_python.sh -m src.cli.train_pipeline --model logistic --output-dir experiments/train

# Support Vector Machine
./scripts/run_python.sh -m src.cli.train_pipeline --model svm --output-dir experiments/train

# Random Forest
./scripts/run_python.sh -m src.cli.train_pipeline --model random_forest --output-dir experiments/train

# XGBoost
./scripts/run_python.sh -m src.cli.train_pipeline --model xgboost --output-dir experiments/train

# BERT (requires GPU)
./scripts/run_python.sh -m src.cli.train_pipeline --model bert --output-dir experiments/train
```

### Hyperparameter Tuning

```bash
# Automated tuning for all models
bash scripts/tune_hyperparams.sh
```

## 📈 Analysis and Visualization

### Generate All Plots

```bash
# Model comparison plots
python scripts/create_analysis_plots.py

# Calibration analysis
python scripts/create_calibration_plots.py

# Error analysis
python scripts/create_error_analysis_plots.py

# Ensemble evaluation
python scripts/create_ensemble_plots.py

# Extended evaluation (out-of-domain)
python scripts/create_extended_evaluation_plots.py
```

### View Results

All visualizations are saved to `reports/` subdirectories:
- **Analysis**: `reports/analysis/` - Model comparisons and correlations
- **Calibration**: `reports/analysis/` - Probability calibration plots
- **Errors**: `reports/error_analysis/` - Error patterns and distributions
- **Ensembles**: `reports/ensemble/` - Ensemble performance analysis
- **Robustness**: `reports/extended_evaluation/` - Out-of-domain evaluation

## 🌐 REST API

Deploy the Flask REST API for programmatic access:

```bash
python scripts/app.py
```

### API Endpoints

- `POST /predict`: Single comment classification
- `POST /batch_predict`: Multiple comments classification
- `GET /models`: List available models
- `GET /health`: Health check

### Example Usage

```python
import requests

response = requests.post("http://localhost:5000/predict",
    json={"text": "This is a toxic comment", "model": "bert"})
print(response.json())
```

## 📊 Experimental Results

### Comprehensive Evaluation

The project includes extensive experimental evaluation:

- **Cross-Validation**: 3-fold temporal splits with 3 random seeds
- **Performance Metrics**: Macro/Micro F1, PR-AUC, ROC-AUC, calibration
- **Per-Label Analysis**: Individual performance across 6 toxicity dimensions
- **Computational Benchmarks**: Training time, inference latency, memory usage
- **Robustness Testing**: Out-of-domain evaluation with synthetic data

### Key Findings

- **BERT Superiority**: Best overall performance (PR-AUC: 0.722) with excellent calibration
- **SVM Reliability**: Strong performance with best calibration among traditional models
- **Rare Label Challenge**: All models struggle with minority classes (threat, identity_hate)
- **Domain Robustness**: BERT maintains performance best on modern internet slang

## 📚 Documentation

### Comprehensive Reports

- **`FINAL_REPORT.md`**: Complete project documentation (560+ pages when compiled)
- **`EXPERIMENTS_REPORT.md`**: Detailed experimental methodology and results
- **`EXPERIMENTS_README.md`**: Experiment execution and analysis guide
- **Model Reports**: Individual detailed reports for Random Forest, SVM, and BERT

### API Documentation

- **`API_README.md`**: Complete API reference and usage examples
- **Demo Guide**: Interactive demo usage instructions

## 🔄 Reproducibility

### Environment Setup

```bash
# Bootstrap complete environment
bash scripts/00_bootstrap_project.sh

# Or skip components if already configured
bash scripts/00_bootstrap_project.sh --skip-env --skip-data
```

### Data Pipeline

```bash
# Download Kaggle dataset
bash scripts/02_download_kaggle.sh

# Generate cross-validation splits
./scripts/run_python.sh -m src.cli.make_splits --folds 3

# Precompute features (optional)
./scripts/run_python.sh -m src.cli.make_normalized_text --data-path data/raw/train.csv
```

## 🛠️ Development

### Adding New Models

1. Create model implementation in `src/models/`
2. Add training logic to `src/pipeline/train.py`
3. Update CLI interface in `src/cli/train_pipeline.py`
4. Add to demo and API if desired

### Extending Analysis

1. Add plotting functions to `scripts/create_*_plots.py`
2. Update analysis in `reports/analysis/`
3. Generate new visualizations automatically

## 📈 Performance Benchmarks

### Training Time (per fold)
- **Logistic Regression**: ~2 minutes
- **SVM**: ~5-15 minutes
- **Random Forest**: ~2-5 minutes
- **XGBoost**: ~3-8 minutes
- **BERT**: ~30-60 minutes (GPU) / 2-4 hours (CPU)

### Inference Latency
- **Traditional Models**: <1ms per comment
- **BERT**: ~5-10ms per comment (GPU) / ~50-100ms (CPU)

### Memory Requirements
- **Training**: 2-8GB RAM (depends on model and batch size)
- **Inference**: <1GB RAM for all models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with comprehensive testing
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Jigsaw/Conversation AI** for the toxic comment classification dataset
- **Hugging Face** for transformer model implementations
- **Scikit-learn** for traditional machine learning algorithms
- **Streamlit** for the interactive demo framework

## 📞 Support

For questions or issues:
- Check the comprehensive documentation in `docs/`
- Review the troubleshooting section in individual README files
- Open an issue on GitHub with detailed information

## 📚 Resources

### Repository
- **GitHub Repository**: [https://github.com/sonalilonkar1/toxic_comments_classification](https://github.com/sonalilonkar1/toxic_comments_classification)

### Project Materials
- **Google Drive Folder**: [Complete Project Resources](https://drive.google.com/drive/folders/1tpX_Saks2ZfG-uSv7rvc0kqVyPfWkIRj?usp=sharing)

This drive contains:
- **Presentation Recordings**: Video recordings of project presentations and demonstrations
- **Slides**: Presentation slides and documentation materials
- **Experiment Results**: Detailed results for all models (TF-IDF variants: Linear Regression, SVM, Random Forest, and BERT)
- **Demo Videos**: Interactive demonstrations showing comment classification with labels, top-K metrics, and explainability features for TF-IDF models

---

**Last Updated**: December 15, 2025
**Version**: 2.0 - Complete ML System
**Models**: 5 architectures fully implemented
**Experiments**: 45+ runs with comprehensive evaluation
**Visualizations**: 25+ automated plots
**Applications**: Web demo + REST API production-ready

