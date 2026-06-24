# AMR ML Project - Complete QA Test Analysis Report

Generated: 2026-05-15  
Role: Senior software tester / quality test officer  
Project: AMR prediction for Acinetobacter baumannii Meropenem resistance

## 1. Executive Summary

This repository is a Python machine-learning project with:

- A CLI pipeline in `main.py`.
- Data ingestion and preprocessing in `controllers/data_controller.py`.
- Model wrappers in `models/amr_models.py`.
- Evaluation logic in `controllers/eval_controller.py`.
- Plot and SHAP view helpers in `views/`.
- A Streamlit dashboard in `streamlit_app/app.py`.
- Notebook generation in `notebooks/generate_notebook.py`.

The project is not currently production-ready in the active environment because the default Python installation is missing required core packages: `xgboost` and `shap`. This blocks the full pipeline, model training, SHAP analysis, and Streamlit dashboard import.

The data-preprocessing path works with the synthetic fallback dataset. Syntax compilation passes. Notebook generation works only when UTF-8 output is forced on Windows.

Overall working stage: prototype / academic demo quality. It can become production-ready after dependency/environment hardening, real data validation, automated tests, model persistence, reproducible environment setup, and dashboard smoke testing.

## 2. Project Working Overview

### CLI entry point

`main.py` supports three commands:

- `python main.py pipeline`: runs data loading, preprocessing, training, evaluation, and SHAP.
- `python main.py download`: downloads BV-BRC CSVs into `data/raw/`.
- `python main.py streamlit`: launches the Streamlit dashboard.

### Data flow

Configured paths are in `config.py`:

- Raw AMR phenotype CSV: `data/raw/amr_phenotype.csv`
- Raw specialty genes CSV: `data/raw/sp_genes.csv`
- Processed feature matrix: `data/processed/feature_matrix.csv`
- Processed gene list: `data/processed/gene_list.txt`

If both raw CSV files exist, `DataController` loads real BV-BRC data. If they do not exist and fallback is enabled, it generates a synthetic BV-BRC-calibrated dataset.

Current local data state:

- `data/raw/` is missing.
- `data/processed/feature_matrix.csv` exists.
- `data/processed/gene_list.txt` exists.
- Current processed matrix shape: 2,500 rows x 51 columns.
- Class distribution: 1,498 resistant and 1,002 susceptible.

### ML flow

The pipeline trains four models:

- XGBoost
- Random Forest
- Gradient Boosting
- Logistic Regression

Evaluation computes accuracy, AUC-ROC, F1, sensitivity, specificity, precision, average precision, ROC/PR curves, and confusion matrices.

SHAP is used for explainability, mainly with XGBoost through `shap.TreeExplainer`.

### Frontend flow

The Streamlit app:

- Loads and preprocesses data using cached resources.
- Trains models in memory.
- Evaluates the test set.
- Computes SHAP explanations.
- Renders pages for overview, dataset exploration, preprocessing, training, evaluation, SHAP, literature comparison, and live prediction.

There is no SQL/NoSQL database server. The "database" dependency is BV-BRC CSV data under `data/raw/`, plus generated files under `data/processed/`.

## 3. Tests and Checks Performed

### Repository structure check

Result: Pass.

Observed key files:

- `main.py`
- `config.py`
- `requirements.txt`
- `controllers/data_controller.py`
- `controllers/train_controller.py`
- `controllers/eval_controller.py`
- `models/amr_models.py`
- `views/plots.py`
- `views/shap_views.py`
- `streamlit_app/app.py`
- `notebooks/generate_notebook.py`
- `README.md`
- `DATA_GUIDE.md`

### Python version check

Command:

```powershell
python --version
```

Result:

```text
Python 3.13.9
```

Risk: README advertises Python 3.11+. The active environment is Python 3.13.9. Some ML libraries may lag behind newer Python versions, so production should pin and test an exact Python version.

### Syntax compilation

Command:

```powershell
python -m compileall .
```

Result: Pass. No syntax errors were reported.

### Dependency availability

Command:

```powershell
python -c "import importlib.util as u; mods=['streamlit','numpy','pandas','matplotlib','seaborn','sklearn','xgboost','shap','plotly','imblearn','nbformat','requests']; print({m: bool(u.find_spec(m)) for m in mods})"
```

Result:

```text
streamlit: True
numpy: True
pandas: True
matplotlib: True
seaborn: True
sklearn: True
xgboost: False
shap: False
plotly: True
imblearn: True
nbformat: True
requests: True
```

Status: Fail. `xgboost` and `shap` are required by the project and missing in the active environment.

### Pipeline execution

Command:

```powershell
python main.py pipeline
```

Result: Fail at model import.

Important output:

```text
BV-BRC CSVs not found - using calibrated synthetic dataset.
Synthetic: 2,500 samples x 48 features
Train: 2,000 | Test: 500
SMOTE -> 2,396 balanced samples
ModuleNotFoundError: No module named 'xgboost'
```

Assessment: Data loading and preprocessing work. Model training cannot start in the active environment.

### Streamlit app import

Command:

```powershell
python -c "import streamlit_app.app"
```

Result: Fail.

Important output:

```text
ModuleNotFoundError: No module named 'shap'
```

Assessment: Dashboard cannot start in the active environment until SHAP is installed.

### Data controller synthetic fallback

Command:

```powershell
python -c "from controllers.data_controller import DataController; dc=DataController(use_synthetic_fallback=True).load().preprocess(); print(dc.data_source, dc.n_samples, len(dc.feature_names), dc.class_balance, dc.label_stats); print(dc.get_splits()[0].shape, dc.get_splits()[1].shape)"
```

Result: Pass.

Observed:

```text
Synthetic (BV-BRC calibrated)
2500 samples
48 features
resistant: 1498
susceptible: 1002
train shape: (2000, 48)
test shape: (500, 48)
```

### MIC parsing edge cases

Command used with UTF-8 output:

```powershell
$env:PYTHONIOENCODING='utf-8'; python -c "from controllers.data_controller import _parse_mic; vals=['16.0','>16','<=2','>=32','>=32','<=2','2 mg/L','4 ug/mL','abc','']; print({v:_parse_mic(v) for v in vals})"
```

Result: Pass for tested values.

Observed:

```text
'16.0': 16.0
'>16': 16.0
'<=2': 2.0
'>=32': 32.0
'2 mg/L': 2.0
'4 ug/mL': 4.0
'abc': None
'': None
```

### Notebook validation/generation

Command:

```powershell
python notebooks/generate_notebook.py
```

Result: Fail in default Windows console encoding.

Error:

```text
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705'
```

Command with UTF-8 output:

```powershell
$env:PYTHONIOENCODING='utf-8'; python notebooks/generate_notebook.py
```

Result: Pass.

Observed:

```text
Notebook written: notebooks/AMR_ML_Project.ipynb
Total cells: 47
20 code, 27 markdown
```

Notebook read also emitted a warning:

```text
MissingIDFieldWarning: Cell is missing an id field
```

After generation, the report command counted missing IDs as 0, so the warning appears during nbformat normalization/validation of the existing notebook state.

### Security pattern scan

Checked for common high-risk patterns:

- Secrets/tokens/passwords
- `eval(`
- `exec(`
- `pickle`
- `joblib`

Result: No obvious hardcoded secrets or direct dynamic execution patterns found.

## 4. Major Findings and Production Weaknesses

### P0 - Active environment cannot run full project

Files:

- `models/amr_models.py`, line 5: `import xgboost as xgb`
- `views/shap_views.py`, line 7: `import shap`
- `streamlit_app/app.py`, line 12: `import shap as shap_lib`

Impact:

- CLI pipeline fails before model training.
- Streamlit dashboard fails at import.
- SHAP explainability is unavailable.

Recommendation:

- Create a clean virtual environment.
- Install `requirements.txt` into that exact environment.
- Verify with `python -c "import xgboost, shap"`.
- Pin dependency versions for reproducibility.

### P0 - Project currently runs on synthetic data, not real BV-BRC data

Files:

- `main.py`, line 15: `DataController(use_synthetic_fallback=True)`
- `streamlit_app/app.py`, line 247: `DataController(use_synthetic_fallback=True)`
- `config.py`, lines 12-24: raw and processed data paths

Impact:

- If raw CSVs are absent, the app silently becomes an educational demo.
- Reported model behavior may not represent real clinical data.
- Production users may trust synthetic results as real performance.

Recommendation:

- For production, disable synthetic fallback by default.
- Show a hard error if real data is missing.
- Clearly label synthetic mode in CLI, dashboard, exported reports, and notebook.

### P1 - No automated unit/integration test suite

Impact:

- Data edge cases, metric calculations, dashboard import, and notebook generation are not protected.
- Future changes can break preprocessing or model behavior silently.

Recommendation:

- Add `pytest`.
- Add tests for MIC parsing, column normalization, missing CSVs, small datasets, duplicate genomes, no-overlap CSVs, class imbalance, metric calculations, and Streamlit import smoke test.

### P1 - Windows console Unicode failures

File:

- `notebooks/generate_notebook.py`, line 592 prints a Unicode check mark.

Impact:

- Notebook generation crashes in default Windows cp1252 terminals.
- Several files contain Unicode and emoji output, so this can affect CLI usage too.

Recommendation:

- Set UTF-8 output in run instructions:

```powershell
$env:PYTHONIOENCODING='utf-8'
```

- Or remove emoji from CLI prints.
- Or configure stdout encoding in `main.py` and notebook generator.

### P1 - Streamlit API compatibility risk

File:

- `streamlit_app/app.py` uses `width='stretch'` heavily, including chart, dataframe, and button calls.

Impact:

- Current environment has Streamlit 1.51.0, where this may work.
- README only requires Streamlit >=1.32.0. Older allowed versions may reject `width='stretch'`.

Recommendation:

- Either pin Streamlit to a version that supports this API, or replace with broadly compatible `use_container_width=True`.

### P1 - No model persistence or versioned artifacts

Impact:

- Models train every run and live only in memory.
- Production cannot reproduce a specific deployed model.
- No artifact metadata for model version, data hash, training timestamp, feature list, or dependency versions.

Recommendation:

- Save trained models with metadata.
- Version `feature_names`.
- Store model metrics beside artifacts.
- Add a loader path for the dashboard to use pre-trained models.

### P1 - Cross-validation leakage/imbalance concern

File:

- `controllers/train_controller.py` intentionally avoids SMOTE inside CV.

Impact:

- Avoiding SMOTE outside CV is good, but when SMOTE is desired for model comparison, the correct pattern is an imbalanced-learn `Pipeline` with SMOTE inside each fold.
- Current CV scores may not represent the same training procedure used for final models.

Recommendation:

- Use `imblearn.pipeline.Pipeline([('smote', SMOTE(...)), ('model', ...)])` inside CV for models evaluated with SMOTE.

### P1 - Clinical safety and regulatory readiness missing

Impact:

- The app predicts antibiotic resistance, which is medically sensitive.
- It includes educational disclaimers, but production would require stronger controls.

Recommendation:

- Add explicit "not for clinical decision making" mode unless validated.
- Add calibration, threshold selection, confidence intervals, subgroup validation, and external validation.
- Add audit logging for predictions if deployed.

### P2 - Generated/cache files in repository working tree

Observed:

- `__pycache__/` directories and `.pyc` files are present.

Impact:

- Noise in repository.
- Can cause confusion across Python versions.

Recommendation:

- Keep `__pycache__/` ignored.
- Remove cache directories before release packaging.

### P2 - README claims and local state are inconsistent

Examples:

- README discusses real BV-BRC results and screenshots.
- Current local run uses synthetic fallback because raw data is absent.

Impact:

- Reviewers may misunderstand whether results are real or synthetic.

Recommendation:

- Add a "Current local data mode" section.
- Keep a reproducible `results/` report generated from the actual run.

## 5. Edge Cases That Should Be Tested Before Production

### Data ingestion edge cases

- Missing `data/raw/` directory.
- Only one of the two CSV files exists.
- Empty CSV files.
- CSV with wrong delimiter.
- CSV with title-case columns.
- CSV with duplicate columns after normalization.
- CSV with no `genome_id`.
- CSV with no `resistant_phenotype`.
- CSV with no `gene`.
- CSV with antibiotic values not equal to `meropenem`.
- CSV with no overlapping genome IDs.
- CSV with less than 10 overlapping genomes.
- CSV with only resistant or only susceptible labels.
- CSV with one sample in a class.
- MIC values containing `>`, `<`, `>=`, `<=`, Unicode greater/less signs, blank strings, non-numeric values, units, and whitespace.
- Intermediate MIC values between susceptible and resistant breakpoints.
- Duplicate genome IDs with conflicting labels.
- Duplicate genome/gene rows.
- Gene names with spaces, slashes, punctuation, or mixed case.
- Very sparse gene matrix.
- Very high-dimensional gene matrix.

### ML edge cases

- Training with no SMOTE installed.
- Training with SMOTE but too few minority samples.
- XGBoost unavailable.
- SHAP unavailable.
- Model predicts a single class.
- `predict_proba` missing or returns unexpected shape.
- ROC-AUC with one class in test set.
- Precision with zero positive predictions.
- Feature importance requested for a model that does not expose `feature_importances_`.
- Logistic regression coefficient display.
- Cross-validation where folds exceed minority class count.

### Frontend edge cases

- App starts with no raw data.
- App starts with real data.
- App starts when generated processed CSVs do not exist.
- All navigation pages render.
- Live prediction with all genes off.
- Live prediction with all genes on.
- Live prediction after presets.
- Live prediction after manual checkbox edits.
- SHAP page with fewer than requested top features.
- Dataset explorer with less than 5 rows.
- Browser refresh and Streamlit cache invalidation.
- Large feature list causing slow checkbox rendering.
- Older Streamlit versions.

### Notebook edge cases

- Generate notebook in Windows PowerShell default encoding.
- Generate notebook with UTF-8 forced.
- Run all cells from project root.
- Run all cells from inside `notebooks/`.
- Missing dependencies in selected Jupyter kernel.

## 6. Recommended Test Suite

Add these files:

- `tests/test_data_controller.py`
- `tests/test_mic_parser.py`
- `tests/test_eval_controller.py`
- `tests/test_model_wrappers.py`
- `tests/test_streamlit_import.py`
- `tests/test_notebook_generator.py`

Minimum acceptance checks:

```powershell
python -m pytest -q
python -m compileall .
python main.py pipeline
python -c "import streamlit_app.app"
$env:PYTHONIOENCODING='utf-8'; python notebooks/generate_notebook.py
```

## 7. Correct Run Instructions

### Step 1 - Create and activate environment

Recommended on Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify:

```powershell
python -c "import xgboost, shap, streamlit, sklearn; print('dependencies ok')"
```

### Step 2 - Add database/data dependency

This project does not use a live database server. It uses BV-BRC CSV files as the data source.

Create this folder:

```powershell
mkdir data\raw
```

Place:

```text
data/raw/amr_phenotype.csv
data/raw/sp_genes.csv
```

Or run:

```powershell
python main.py download
```

### Step 3 - Run backend pipeline first

```powershell
python main.py pipeline
```

Expected behavior:

- Load real CSVs if present.
- Otherwise use synthetic fallback.
- Preprocess and save `data/processed/feature_matrix.csv`.
- Train four models.
- Evaluate metrics.
- Compute SHAP values.

Production recommendation: disable synthetic fallback for real deployment.

### Step 4 - Run frontend dashboard

```powershell
streamlit run streamlit_app/app.py
```

Or:

```powershell
python main.py streamlit
```

Expected URL:

```text
http://localhost:8501
```

The Streamlit app retrains in memory through cached resources. If raw CSVs are updated, clear Streamlit cache or restart the app.

### Step 5 - Windows UTF-8 setting for notebook/CLI stability

Before notebook generation on Windows PowerShell:

```powershell
$env:PYTHONIOENCODING='utf-8'
python notebooks/generate_notebook.py
```

## 8. Production Readiness Checklist

Required before industry use:

- Resolve active Python environment dependency mismatch.
- Pin exact dependency versions.
- Add automated tests.
- Add CI pipeline.
- Disable synthetic fallback in production.
- Add real BV-BRC data validation.
- Add model artifact saving/loading.
- Add model/data versioning.
- Add calibration and threshold validation.
- Add external validation dataset.
- Add dashboard smoke tests.
- Add logging and clear error handling.
- Remove generated cache files from release.
- Clarify medical disclaimer and non-clinical status.

## 9. Final QA Verdict

Current stage: functional prototype, not production-ready.

Backend data preprocessing is working. Full ML training and explainability are blocked in the active environment because `xgboost` and `shap` are missing. The frontend is also blocked by missing SHAP. The project architecture is understandable and close to a complete academic ML demo, but production readiness requires dependency cleanup, real data enforcement, automated tests, reproducible model artifacts, and stronger validation.
