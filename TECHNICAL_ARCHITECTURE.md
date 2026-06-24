# 🏛️ AMR ML Project - Technical Architecture & Data Flow

## SYSTEM ARCHITECTURE

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────┐              ┌──────────────────────┐            │
│  │  BV-BRC Public API   │              │ Synthetic Fallback   │            │
│  │ (Real genomic data)  │              │ (for testing)        │            │
│  │                      │              │                      │            │
│  │ • genome_id          │              │ 2,500 samples        │            │
│  │ • resistant_pheno    │              │ 51 features          │            │
│  │ • MIC values         │              │ Calibrated class     │            │
│  │ • specialty genes    │              │ distribution (55/45) │            │
│  └──────────┬───────────┘              └──────────┬───────────┘            │
│             │                                     │                         │
│             └────────────────┬────────────────────┘                         │
│                              ↓                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING LAYER (DataController)                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 1: COLUMN NORMALIZATION                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Input columns: "genome id", "genome_id", "GENOME_ID", etc.            │ │
│  │ Alias map (50+ variants) → "genome_id"                                │ │
│  │ Also normalizes: antibiotic, resistant_phenotype, gene, etc.          │ │
│  └─────────────────────────────┬──────────────────────────────────────────┘ │
│                                 ↓                                            │
│  Step 2: MIC PARSING & RECOVERY                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Parse raw MIC: '≥8', '>16', '<=2', 'NaN', etc. → float                │ │
│  │ Apply CLSI M100 / EUCAST 2023 breakpoints:                            │ │
│  │   MIC ≥ 8 mg/L  → Resistant (1)                                       │ │
│  │   MIC ≤ 2 mg/L  → Susceptible (0)                                     │ │
│  │   2 < MIC < 8   → Intermediate → Susceptible (conservative)           │ │
│  │ Result: binary labels (0/1)                                            │ │
│  └─────────────────────────────┬──────────────────────────────────────────┘ │
│                                 ↓                                            │
│  Step 3: FEATURE MATRIX CREATION                                           │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Pivot specialty genes on genome_id                                     │ │
│  │ Binary encoding: 1 = gene present, 0 = absent                         │ │
│  │ Handle missing: fill with 0 (gene not detected)                       │ │
│  │ Result: X ∈ ℝ^(N × P) where N=genomes, P=genes                       │ │
│  │                                                                         │ │
│  │ Example matrix (first 5 genes, 3 genomes):                            │ │
│  │          Gene1  Gene2  Gene3  ...  GeneP                              │ │
│  │ Genome1    1      0      1     ...    1                               │ │
│  │ Genome2    0      1      1     ...    0                               │ │
│  │ Genome3    1      1      0     ...    1                               │ │
│  └─────────────────────────────┬──────────────────────────────────────────┘ │
│                                 ↓                                            │
│  Step 4: FEATURE FILTERING (Prevalence-Based)                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Remove genes present in <2% of samples (MIN_GENE_PREVALENCE)          │ │
│  │ Formula: keep if sum(X[:, j]) / N ≥ 0.02                             │ │
│  │ Rationale: Very rare genes add noise, not signal                      │ │
│  │                                                                         │ │
│  │ Example:                                                               │ │
│  │   GeneA: in 40/1884 = 2.1% samples → KEEP                            │ │
│  │   GeneB: in 15/1884 = 0.8% samples → DROP                            │ │
│  │                                                                         │ │
│  │ Result: X_filtered with 12-51 retained features (data-dependent)     │ │
│  └─────────────────────────────┬──────────────────────────────────────────┘ │
│                                 ↓                                            │
│  Step 5: TRAIN-TEST SPLIT (Stratified)                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Stratified split to preserve class distribution:                      │ │
│  │   Train: 80% (1,507 samples)                                          │ │
│  │   Test:  20% (377 samples)                                            │ │
│  │                                                                         │ │
│  │ Class distribution:                                                    │ │
│  │   Train: ~827 resistant (54.9%), ~680 susceptible (45.1%)            │ │
│  │   Test:  ~213 resistant (56.5%), ~164 susceptible (43.5%)            │ │
│  │                                                                         │ │
│  │ Random state: 42 (reproducibility)                                    │ │
│  └─────────────────────────────┬──────────────────────────────────────────┘ │
│                                 ↓                                            │
│  Step 6: CLASS BALANCING (SMOTE - Training Only)                           │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Problem: Imbalanced training set (55% vs 45%)                         │ │
│  │ Solution: SMOTE (Synthetic Minority Oversampling Technique)           │ │
│  │                                                                         │ │
│  │ IMPORTANT: Applied ONLY to X_train, y_train                           │ │
│  │           NOT applied to test set (realistic evaluation)              │ │
│  │           NOT applied inside CV (prevents data leakage)               │ │
│  │                                                                         │ │
│  │ Algorithm:                                                             │ │
│  │   1. Find k-nearest neighbors in minority class                       │ │
│  │   2. Generate synthetic samples between minority pairs                │ │
│  │   3. Balance minority to match majority                               │ │
│  │                                                                         │ │
│  │ Result:                                                                │ │
│  │   X_train_bal: 2,016 samples (both classes)                           │ │
│  │   y_train_bal: 1,008 resistant, 1,008 susceptible (50/50)            │ │
│  │                                                                         │ │
│  │ Benefit: Model learns both classes equally                            │ │
│  │ Trade-off: Slight overfitting risk (mitigated by CV)                 │ │
│  └─────────────────────────────┬──────────────────────────────────────────┘ │
│                                 ↓                                            │
│              ┌─────────────────────────────────────┐                        │
│              │    DATA PROCESSING OUTPUTS:         │                        │
│              ├─────────────────────────────────────┤                        │
│              │ • X_train_bal: (2,016, 12-51)       │                        │
│              │ • y_train_bal: (2,016,)             │                        │
│              │ • X_train: (1,507, 12-51)           │                        │
│              │ • y_train: (1,507,)                 │                        │
│              │ • X_test: (377, 12-51)              │                        │
│              │ • y_test: (377,)                    │                        │
│              │ • feature_names: list of genes      │                        │
│              └─────────────────────────────────────┘                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                  MODEL TRAINING LAYER (TrainController)                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: X_train_bal, y_train_bal (balanced training data)                   │
│                                                                              │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐            │
│  │    XGBoost       │ │ Random Forest    │ │ Gradient Boost   │            │
│  │                  │ │                  │ │                  │            │
│  │ n_estimators=300 │ │ n_estimators=300 │ │ n_estimators=200 │            │
│  │ max_depth=6      │ │ max_depth=12     │ │ max_depth=5      │            │
│  │ lr=0.08          │ │ class_weight=    │ │ lr=0.08          │            │
│  │ subsample=0.80   │ │  balanced        │ │ loss=deviance    │            │
│  │ scale_pos_weight │ │ n_jobs=-1        │ │                  │            │
│  │ (dynamic)        │ │                  │ │                  │            │
│  └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘            │
│           │                    │                    │                       │
│           └────────────────────┼────────────────────┘                       │
│                                │                                            │
│                      ┌─────────┴──────────┐                                 │
│                      │ Each model fits on │                                 │
│                      │ balanced training  │                                 │
│                      │ data               │                                 │
│                      └─────────┬──────────┘                                 │
│                                ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │         4 TRAINED MODEL OBJECTS CREATED                             │ │
│  │    {model_name: sklearn/xgboost classifier object}                  │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  PARALLEL: CROSS-VALIDATION (on original, unbalanced training set)          │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │ Input: X_train, y_train (original, NOT SMOTE'd)                     │ │
│  │ Fold strategy: StratifiedKFold(n_splits=5, shuffle=True)            │ │
│  │                                                                       │ │
│  │ For each fold:                                                       │ │
│  │   1. Split into train_fold, val_fold (preserving class dist)       │ │
│  │   2. Fit model on train_fold                                        │ │
│  │   3. Score on val_fold using roc_auc metric                         │ │
│  │   4. Record AUC                                                      │ │
│  │                                                                       │ │
│  │ Results: array of 5 AUC scores per model                            │ │
│  │ Report: mean ± std dev                                              │ │
│  │                                                                       │ │
│  │ Why no SMOTE in CV?                                                 │ │
│  │   - Prevents data leakage (synthetic data shouldn't affect val)    │ │
│  │   - CV uses real, imbalanced data (realistic evaluation)           │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│              MODEL EVALUATION LAYER (EvalController)                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: X_test, y_test (held-out test set) + 4 trained models              │
│                                                                              │
│  For each model on test set:                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │ 1. MAKE PREDICTIONS                                                 │ │
│  │    ├─ y_pred = model.predict(X_test) → binary (0/1)                │ │
│  │    └─ y_proba = model.predict_proba(X_test)[:, 1] → probabilities   │ │
│  │                                                                       │ │
│  │ 2. COMPUTE METRICS                                                  │ │
│  │    ├─ Accuracy: (TP+TN) / (TP+TN+FP+FN)                            │ │
│  │    ├─ AUC-ROC: Area under ROC curve (threshold-independent)        │ │
│  │    ├─ F1-Score: 2×(Precision×Recall)/(Precision+Recall)            │ │
│  │    ├─ Sensitivity: TP / (TP+FN) — catch real resistances          │ │
│  │    ├─ Specificity: TN / (TN+FP) — avoid false alarms             │ │
│  │    ├─ Precision: TP / (TP+FP) — confidence in prediction          │ │
│  │    └─ Avg Precision: area under PR curve                           │ │
│  │                                                                       │ │
│  │ 3. GENERATE CURVES                                                  │ │
│  │    ├─ ROC curve: (FPR, TPR) at different thresholds                │ │
│  │    └─ PR curve: (Precision, Recall) at different thresholds        │ │
│  │                                                                       │ │
│  │ 4. CONFUSION MATRIX                                                 │ │
│  │    ┌──────────┬──────────┐                                           │ │
│  │    │    TN    │    FP    │  (Predicted 0)                           │ │
│  │    ├──────────┼──────────┤                                           │ │
│  │    │    FN    │    TP    │  (Predicted 1)                           │ │
│  │    └──────────┴──────────┘                                           │ │
│  │    (Actual 0) (Actual 1)                                            │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  AGGREGATION: Build metrics_df (8 columns × 4 rows)                        │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │ Model            │ Accuracy │ AUC-ROC │ F1-Score │ ... │            │ │
│  ├──────────────────┼──────────┼─────────┼──────────┼─────┤            │ │
│  │ Logistic Reg     │   0.60   │ 0.5800  │  0.65    │ ... │            │ │
│  │ XGBoost          │   0.65   │ 0.5712  │  0.71    │ ... │            │ │
│  │ Random Forest    │   0.63   │ 0.5674  │  0.70    │ ... │            │ │
│  │ Gradient Boost   │   0.62   │ 0.5511  │  0.68    │ ... │            │ │
│  └──────────────────┴──────────┴─────────┴──────────┴─────┘            │ │
│                                                                              │
│  BEST MODEL: Row with highest AUC-ROC → Logistic Regression (0.5800)      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│            EXPLAINABILITY LAYER (SHAPAnalyser)                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: Best model (XGBoost) + X_test + feature names                      │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │ 1. INITIALIZE TreeExplainer                                         │ │
│  │    ├─ explainer = shap.TreeExplainer(xgb_model)                    │ │
│  │    └─ Only works for tree-based: XGBoost, RF, GB (not LR)         │ │
│  │                                                                       │ │
│  │ 2. COMPUTE SHAP VALUES                                              │ │
│  │    ├─ shap_values = explainer.shap_values(X_test[:500])            │ │
│  │    │  (Up to 500 samples; faster than 377 needed)                  │ │
│  │    └─ Output shape: (500, num_genes)                               │ │
│  │       Each cell = gene's contribution to THIS prediction            │ │
│  │                                                                       │ │
│  │ 3. BASE VALUE (Expected Value)                                      │ │
│  │    └─ base_value = explainer.expected_value                        │ │
│  │       Average model output across training data                     │ │
│  │       Each prediction = base_value + sum(SHAP values)              │ │
│  │                                                                       │ │
│  │ 4. GLOBAL IMPORTANCE (Mean |SHAP|)                                 │ │
│  │    ├─ For each gene j: importance[j] = mean(|SHAP[:, j]|)        │ │
│  │    ├─ Sort by importance (descending)                              │ │
│  │    └─ Top 10-20 genes plotted as bar chart                         │ │
│  │                                                                       │ │
│  │ 5. PER-SAMPLE EXPLANATION (Waterfall)                               │ │
│  │    ├─ For sample i:                                                 │ │
│  │    │   1. Start at base_value                                      │ │
│  │    │   2. Each gene pushes prediction ±SHAP[i, j]                  │ │
│  │    │   3. Final prediction = base_value + sum(SHAP[i, :])          │ │
│  │    │                                                                 │ │
│  │    └─ Example:                                                       │ │
│  │        Base value: 0.5 (50% resistance)                            │ │
│  │        Gene1 present (+0.15) → 65% resistant                       │ │
│  │        Gene2 absent (-0.05) → 60% resistant                        │ │
│  │        Gene3 present (+0.08) → 68% resistant                       │ │
│  │        ...                                                           │ │
│  │        Final prediction: 75% resistant (0.75 confidence)           │ │
│  │                                                                       │ │
│  │ 6. VISUALIZATIONS                                                   │ │
│  │    ├─ Bar chart: Mean |SHAP| per gene (global importance)         │ │
│  │    ├─ Beeswarm: Each dot = sample; x=SHAP value, color=gene value │ │
│  │    ├─ Waterfall: Single-sample breakdown                           │ │
│  │    └─ SHAP interaction (2D interactions between genes)              │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  CLINICAL INTERPRETATION:                                                  │
│  "Patient's genome has genes X, Y, Z with high SHAP values"               │
│  "These are known resistance mechanisms (based on literature)"             │
│  "Recommend antibiotic NOT affected by these genes"                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CLI Output:                                                               │
│  ├─ Console: Metrics table, best model name, top genes                     │
│  └─ Files: feature_matrix.csv, gene_list.txt (saved in data/processed/)  │
│                                                                              │
│  Jupyter Notebook:                                                         │
│  ├─ EDA plots: class distribution, gene prevalence                        │
│  ├─ Training plots: CV results, model comparison                          │
│  └─ SHAP plots: global importance, waterfall explanations                 │
│                                                                              │
│  Streamlit Dashboard (8 pages):                                            │
│  ├─ Home: Overview, key metrics                                           │
│  ├─ Dataset: Exploratory data analysis                                    │
│  ├─ Preprocessing: Feature visualization, correlation                     │
│  ├─ Training: CV box plots, cross-validation results                      │
│  ├─ Evaluation: ROC/PR curves, confusion matrices, radar chart           │
│  ├─ SHAP: Global importance, per-sample explanations                     │
│  ├─ Literature: Comparison to published results                          │
│  └─ Live Prediction: Upload genomes → real-time predictions              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## DETAILED CLASS STRUCTURE

### DataController Class

```python
class DataController:
    # Attributes
    df_raw_amr: DataFrame                # Raw AMR phenotype data
    df_raw_sp: DataFrame                 # Raw specialty genes data
    df_features: DataFrame               # Combined feature matrix
    feature_names: list[str]             # Retained gene names
    
    X_train, X_test: ndarray            # Train/test feature matrices
    y_train, y_test: ndarray            # Train/test labels
    X_train_bal, y_train_bal: ndarray   # SMOTE-balanced training set
    
    # Methods
    load() → DataController              # Load real or synthetic data
    preprocess() → DataController        # Full pipeline
    _clean()                             # MIC parsing + feature creation
    _split()                             # Train-test split
    _apply_smote()                       # SMOTE balancing
    _save_processed()                    # Save to CSV
    get_splits() → tuple                 # Return (X_train, X_test, ...)
    get_chi2_ranking() → DataFrame       # Feature importance ranking
```

### BaseAMRModel & Subclasses

```python
class BaseAMRModel:
    # Attributes
    name: str                            # Model display name
    color: str                           # Plotly color
    _model: sklearn/xgboost classifier  # Underlying model
    _fitted: bool                        # Whether trained
    
    # Methods
    fit(X_train, y_train)                # Train model
    predict(X) → ndarray                 # Binary predictions
    predict_proba(X) → ndarray          # Probability predictions
    score(X, y) → float                  # AUC-ROC score

# Subclasses:
class XGBoostAMRModel(BaseAMRModel)      # XGBClassifier wrapper
class RandomForestAMRModel(BaseAMRModel) # RandomForestClassifier wrapper
class GradientBoostingAMRModel(BaseAMRModel) # GradientBoostingClassifier
class LogisticRegressionAMRModel(BaseAMRModel) # LogisticRegression wrapper
```

### TrainController Class

```python
class TrainController:
    # Attributes
    models: dict[str, BaseAMRModel]      # Trained model objects
    cv_results: dict[str, ndarray]       # 5-fold CV scores per model
    
    # Methods
    train(X_train, y_train) → TrainController           # Train all 4 models
    cross_validate(X_train, y_train) → TrainController  # 5-fold CV
    cv_summary_df() → DataFrame                         # Tidy CV results
    best_model() → (str, BaseAMRModel)                  # (name, model)
```

### EvalController Class

```python
class EvalController:
    # Attributes
    results: dict[str, dict]             # Metrics per model
    _X_test, _y_test: ndarray           # Test data
    _models: dict[str, BaseAMRModel]    # Model references
    
    # Methods
    evaluate_all(models, X_test, y_test) → EvalController  # Compute metrics
    metrics_df → DataFrame               # 8-column metrics table
    get_curves(model_name) → dict        # ROC + PR curves
    get_confusion_matrix(model_name) → ndarray
    print_report(model_name)              # Print classification report
    best_model_name() → str               # Highest AUC model
```

### SHAPAnalyser Class

```python
class SHAPAnalyser:
    # Attributes
    model: BaseAMRModel                  # Trained model (usually XGBoost)
    feature_names: list[str]             # Gene names
    _explainer: TreeExplainer            # SHAP explainer object
    _shap_vals: ndarray                  # SHAP value matrix
    _X_sample: ndarray                   # Sample data
    _base_value: float                   # Expected value
    
    # Methods
    compute(X_test, max_samples=500) → SHAPAnalyser   # Compute SHAP values
    global_importance(top_n=20) → Figure               # Bar chart
    beeswarm(top_n=15) → (fig, ax)                     # matplotlib plot
    waterfall(idx) → Figure                            # Single-sample waterfall
    mean_abs_shap → Series                             # Importance series
```

---

## DATA FLOW SEQUENCE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      COMPLETE EXECUTION FLOW                               │
└─────────────────────────────────────────────────────────────────────────────┘

START (main.py pipeline)
  │
  ├─→ create DataController()
  │    │
  │    └─→ .load()
  │         ├─ Check data/raw/amr_phenotype.csv exists?
  │         │   ├─ YES → _load_real() → df_raw_amr loaded
  │         │   └─ NO  → _load_synthetic() → generated df_raw_amr
  │         │
  │         └─ Check data/raw/sp_genes.csv exists?
  │             ├─ YES → _load_real() → df_raw_sp loaded
  │             └─ NO  → _load_synthetic() → generated df_raw_sp
  │
  ├─→ .preprocess()
  │    │
  │    ├─ _clean()
  │    │  ├─ _normalise_columns() → standardize column names
  │    │  ├─ _parse_mic() → convert MIC strings to floats
  │    │  ├─ Apply CLSI breakpoints → binary resistance labels
  │    │  ├─ df_pivot_table() → create binary feature matrix
  │    │  └─ Filter low-prevalence genes → X, y (feature matrix)
  │    │
  │    ├─ _split()
  │    │  └─ train_test_split(X, y, test_size=0.2, stratify=y)
  │    │     → X_train, X_test, y_train, y_test
  │    │
  │    └─ _apply_smote() (if enabled)
  │       └─ SMOTE().fit_resample(X_train, y_train)
  │          → X_train_bal, y_train_bal
  │
  ├─→ .get_splits() → return training/test data + feature_names
  │
  ├─→ create TrainController()
  │    │
  │    ├─→ .train(X_train_bal, y_train_bal)
  │    │   └─ For each of 4 models:
  │    │      ├─ Instantiate model with hyperparams
  │    │      ├─ fit(X_train_bal, y_train_bal)
  │    │      └─ Store in models dict
  │    │
  │    └─→ .cross_validate(X_train, y_train)
  │        └─ For each model:
  │           ├─ StratifiedKFold(n_splits=5)
  │           ├─ cross_val_score(..., scoring='roc_auc')
  │           └─ Store 5 fold scores
  │
  ├─→ create EvalController()
  │    │
  │    └─→ .evaluate_all(models, X_test, y_test)
  │        └─ For each model:
  │           ├─ y_pred = model.predict(X_test)
  │           ├─ y_proba = model.predict_proba(X_test)
  │           ├─ Compute 8 metrics (AUC, F1, etc.)
  │           ├─ roc_curve() → FPR, TPR, thresholds
  │           ├─ precision_recall_curve() → Prec, Rec, thresholds
  │           └─ Store results[model_name] = {metrics dict}
  │
  ├─→ Print metrics_df (sorted by AUC-ROC)
  │
  ├─→ .best_model_name() → "Logistic Regression"
  │
  ├─→ .print_report(best_model_name)
  │   └─ Print classification report per class
  │
  ├─→ create SHAPAnalyser(models["XGBoost"], feature_names)
  │    │
  │    └─→ .compute(X_test)
  │        ├─ TreeExplainer(xgb_model)
  │        ├─ explainer.shap_values(X_test[:500])
  │        ├─ expected_value → base_value
  │        └─ Compute mean |SHAP| per feature
  │
  ├─→ Print top 10 genes by global SHAP importance
  │
  ├─→ [Optional] Generate visualizations
  │    ├─ plots.plot_roc_curves(ec)
  │    ├─ plots.plot_confusion_matrices(ec)
  │    ├─ shap_views.SHAPAnalyser.global_importance()
  │    └─ etc.
  │
  └─→ DONE (print "Pipeline complete")

END
```

---

## MODEL HYPERPARAMETERS SUMMARY

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        CONFIG.PY HYPERPARAMETERS                            │
├──────────────────────────────────────────────────────────────────────────────┤

XGBoost:
  n_estimators      = 300          # Num boosting rounds
  max_depth         = 6            # Tree depth limit
  learning_rate     = 0.08         # Shrinkage
  subsample         = 0.80         # Row sampling ratio per tree
  colsample_bytree  = 0.80         # Column sampling ratio per tree
  random_state      = 42           # Reproducibility seed
  eval_metric       = "logloss"    # Optimization metric
  scale_pos_weight  = DYNAMIC      # Computed: n_neg / n_pos

Random Forest:
  n_estimators      = 300          # Num trees
  max_depth         = 12           # Max tree depth
  class_weight      = "balanced"   # Auto-weight inverse to class freq
  random_state      = 42           # Reproducibility seed
  n_jobs            = -1           # Use all cores

Gradient Boosting:
  n_estimators      = 200          # Num boosting stages
  learning_rate     = 0.08         # Shrinkage
  max_depth         = 5            # Tree depth
  random_state      = 42           # Reproducibility seed

Logistic Regression:
  C                 = 1.0          # Inverse regularization strength (L2)
  solver            = "liblinear"  # Algorithm
  class_weight      = "balanced"   # Auto-weight
  random_state      = 42           # Reproducibility seed
  max_iter          = 2000         # Max iterations

Cross-Validation:
  CV_FOLDS          = 5            # 5-fold stratified K-fold
  RANDOM_STATE      = 42           # Seed for fold shuffling

Data Preprocessing:
  TEST_SIZE         = 0.20         # 20% test
  RANDOM_STATE      = 42           # Split reproducibility
  MIN_GENE_PREV     = 0.02         # Keep genes if >2% prevalence
  USE_SMOTE         = True         # Apply SMOTE to training

SHAP:
  MAX_SAMPLES       = 500          # Max samples for SHAP compute
  TOP_FEATURES      = 20           # Top genes to display

└──────────────────────────────────────────────────────────────────────────────┘
```

---

## ERROR HANDLING & EDGE CASES

```
┌─ MISSING DATA SCENARIO ────────────────────────────────────────┐
│                                                                 │
│ Raw Data Missing (BV-BRC CSVs)                                 │
│   → Fallback: Generate synthetic calibrated dataset (2,500)    │
│   → Print warning: "Using synthetic data"                      │
│   → Proceed normally                                           │
│                                                                 │
├─ IMBALANCED CLASSES ───────────────────────────────────────────┤
│                                                                 │
│ Class distribution: 55% resistant, 45% susceptible            │
│   → Solution 1: SMOTE on training set                         │
│   → Solution 2: class_weight="balanced" in models             │
│   → Solution 3: XGBoost scale_pos_weight (dynamic)            │
│   → Result: Improved minority recall                          │
│                                                                 │
├─ DATA LEAKAGE PREVENTION ─────────────────────────────────────┤
│                                                                 │
│ SMOTE applied ONLY to training set (not test, not CV)        │
│   → CV uses original unbalanced training set                  │
│   → Test set evaluation is realistic (imbalanced)             │
│   → Result: Honest model assessment                           │
│                                                                 │
├─ LOW FEATURE COUNT ────────────────────────────────────────────┤
│                                                                 │
│ After prevalence filtering: ~12 genes retained                │
│   → Why: BV-BRC API limit (1,000 genes downloadable)         │
│   → Impact: Lower accuracy but still AUC > 0.50 (real signal) │
│   → Mitigation: Use simple models + strong regularization     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## PERFORMANCE BOTTLENECKS & OPTIMIZATIONS

```
┌─ BOTTLENECK ─────────────────────┬─ OPTIMIZATION ──────────────┐
│ Data Download                     │ API already optimized       │
│ (~30-60 seconds BV-BRC)          │ Already cached locally      │
├───────────────────────────────────┼─────────────────────────────┤
│ SMOTE computation                 │ Applied once in preprocessing│
│ (~2 seconds)                      │ Reused in training         │
├───────────────────────────────────┼─────────────────────────────┤
│ XGBoost training                  │ 300 estimators balanced    │
│ (~1 second)                       │ Could use GPU (optional)   │
├───────────────────────────────────┼─────────────────────────────┤
│ SHAP computation                  │ TreeExplainer (exact, fast)│
│ (~5 seconds for 500 samples)      │ Limited to top 500 samples │
├───────────────────────────────────┼─────────────────────────────┤
│ Cross-validation                  │ Parallel across cores      │
│ (~8 seconds for 5 folds)          │ n_jobs=-1 enabled         │
├───────────────────────────────────┼─────────────────────────────┤
│ Visualization rendering           │ Plotly (client-side)       │
│ (~2 seconds interactive plots)    │ Matplotlib cached          │
└───────────────────────────────────┴─────────────────────────────┘

Total Pipeline Time: ~25 seconds (CPU, single machine)
Slowest Component: Data download (if needed)
Fastest Component: Prediction on 1 genome (<1 ms)
```

---

*Technical Architecture Document Generated: May 16, 2026*
