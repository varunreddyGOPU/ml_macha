# Release Notes

## v1.0.0 — Initial Release (2026-02-28)

### Highlights

**ML Macha** is a comprehensive Kubeflow Pipelines (KFP) library that provides
reusable, composable pipeline components for building end-to-end ML workflows.

### Features

#### Frameworks Supported
- **Scikit-learn** — classification & regression with full hyperparameter tuning
- **XGBoost** — gradient boosted trees with early-stopping and GPU support
- **Keras** — high-level deep learning with custom architectures
- **TensorFlow** — SavedModel export, TFLite conversion, distributed training
- **PyTorch** — custom `nn.Module` training with learning-rate scheduling
- **AutoML (FLAML)** — automatic model selection and hyperparameter optimization

#### Pipeline Components
- **Data Preparation** — ingestion (GCS, BigQuery, CSV, Parquet), validation, transformation, feature engineering
- **Training** — framework-agnostic trainer, Optuna-based hyperparameter tuning, generic trainer component
- **Evaluation** — metric computation (classification, regression, ranking), model blessing gate
- **Deployment** — blue/green, canary, rolling, and shadow deployment strategies
- **Monitoring** — drift detection (PSI, KS, Wasserstein, Chi²), alerting (Slack, email)
- **Container** — Kaniko/Cloud Build image building, Artifact Registry management

#### Pre-built Pipelines
- `create_training_pipeline` — data → train → evaluate
- `create_deployment_pipeline` — build → deploy → validate
- `create_monitoring_pipeline` — drift → alert → log
- `create_full_ml_pipeline` — end-to-end: data → deploy → notify

#### Configuration
- Dataclass-based configs for compute resources, training, evaluation, and monitoring
- Pre-built compute profiles: `SMALL_CPU`, `MEDIUM_CPU`, `LARGE_CPU`, `SMALL_GPU`, `LARGE_GPU`
- GPU, node selector, and toleration support

#### DevOps
- GitHub Actions CI/CD — lint, test, build wheel, publish to PyPI, create GitHub Release
- Separate workflows for setuptools, Poetry, and uv
- `.gitignore` covering Python, IDE, OS, Docker, testing artifacts
- Dockerfile for containerized execution

#### Documentation
- `README.md` — project overview, installation, quick-start examples
- `use.md` — comprehensive per-file usage guide with 5 end-to-end scenarios
- `BUILDING.md` — wheel building, GitHub setup, Poetry guide, uv guide

### Installation

```bash
pip install kfp-ml-library                     # core
pip install "kfp-ml-library[tensorflow]"       # + TensorFlow
pip install "kfp-ml-library[pytorch]"          # + PyTorch
pip install "kfp-ml-library[gcp]"              # + GCP services
pip install "kfp-ml-library[all]"              # everything
pip install "kfp-ml-library[dev]"              # dev tools
```

### Known Limitations
- Tests directory is scaffolded but tests are not yet implemented
- `auto-sklearn` extra requires Linux (not available on Windows/macOS)
- Entry-point `kfp-ml` is a placeholder (prints version only)

### Dependencies
- Python ≥ 3.9
- kfp ≥ 2.5.0, pandas ≥ 2.1.0, numpy ≥ 1.26.0, scikit-learn ≥ 1.3.0
- Full list in `requirements.txt`

---

*Repository: https://github.com/varunreddyGOPU/ml_macha*
