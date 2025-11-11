# Detailed Implementation Plan: Iris ML Demo Notebook

## Overview

Create a Jupyter notebook at `/simple_ml/iris.ipynb` that demonstrates MLflow logging, automated model selection (automl), and hyperparameter tuning with Hyperopt. The notebook will train and tune a multi-class classifier on the Iris dataset to produce an optimal model for demonstration purposes.

The notebook should be split into multiple cells logically and must be easy to read / follow. Be sure to document the process as you build such that it looks somewhat like a tutorial.

## Context & Environment

- **Date Context**: November 11, 2025 - all documentation references should be recent
- **Environment**: Databricks Connect (local driver, live Databricks connection).
  - conda env is py312dbx
- **Dataset**: Unity Catalog table `main.tomes_gen.iris`
- **Target Variable**: `Species` (multi-class classification with 3 classes)

## Step-by-Step Implementation Plan

### Phase 1: Setup and Data Preparation

#### Step 1.1: Import Required Libraries

Create the first cell with all necessary imports:

- System imports: `sys`, `os` (with path setup: `sys.path.append(os.path.abspath(".."))`)
- Spark imports: `from spark_env import spark`, `from pyspark.sql.functions import col, lit`, `import pyspark.sql.functions as F`
- MLflow: `import mlflow`
- Databricks AutoML: `import databricks.automl`
- Scikit-learn (needed for Hyperopt tuning in Phase 4):
  - `from sklearn.model_selection import train_test_split`
  - `from sklearn.svm import SVC`
  - `from sklearn.linear_model import LogisticRegression`
  - `from sklearn.tree import DecisionTreeClassifier`
  - `from sklearn.ensemble import RandomForestClassifier`
  - `from sklearn.metrics import accuracy_score, classification_report, confusion_matrix`
  - `from sklearn.preprocessing import LabelEncoder` (if needed for target encoding)
- LightGBM (needed for Hyperopt tuning in Phase 4):
  - `import lightgbm as lgb`
  - `from lightgbm import LGBMClassifier`
- Hyperopt:
  - `from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials`
  - `from hyperopt.pyll import scope`
- NumPy and Pandas: `import numpy as np`, `import pandas as pd`

#### Step 1.2: Load Data from Unity Catalog

- Load the dataset: `df = spark.table("main.tomes_gen.iris")`
- Display basic info: `.show()`, `.printSchema()`, `.describe().show()`
- Verify target column name (`Species`) and class distribution
- Note: Databricks AutoML can work with Spark DataFrames directly, but can also accept Pandas DataFrames or table names as strings

#### Step 1.3: Data Preprocessing (Optional for AutoML)

- Databricks AutoML handles most preprocessing automatically, but verify:
  - Target column is correctly identified (`Species`)
  - Exclude `Id` column if present (use `exclude_cols` parameter in AutoML)
  - Check for any missing values (AutoML will handle imputation automatically)
- Note: AutoML will automatically split the data into train/validation/test sets

### Phase 2: MLflow Configuration

#### Step 2.1: Initialize MLflow

- Set tracking URI: `mlflow.set_tracking_uri("databricks")`
- Set experiment: `mlflow.set_experiment("iris_demo")`
- Enable autologging: `mlflow.sklearn.autolog()`
- Verify connection and experiment creation

#### Step 2.2: Document MLflow Setup

- Add markdown cell explaining MLflow configuration
- Note that autolog will automatically track:
  - Parameters
  - Metrics
  - Model artifacts
  - Model signatures

### Phase 3: Databricks AutoML Model Selection

#### Step 3.1: Configure AutoML Experiment

- Set up AutoML to use the same MLflow experiment (optional, AutoML will create its own if not specified):
  - AutoML will automatically create MLflow runs for each trial
  - Consider setting `experiment_name="iris_demo"` to align with Phase 2 experiment, or let AutoML create a new experiment
- Note: AutoML automatically logs all runs to MLflow, so MLflow autologging from Phase 2 will work in conjunction

#### Step 3.2: Run Databricks AutoML Classification

- Call `databricks.automl.classify()` with the following parameters:
  ```python
  summary = databricks.automl.classify(
      dataset=df,  # Can be Spark DataFrame, Pandas DataFrame, or table name string
      target_col="Species",
      primary_metric="f1",  # Options: "f1", "log_loss", "precision", "accuracy", "roc_auc"
      experiment_name="iris_demo",  # Optional: align with Phase 2 experiment
      exclude_cols=["Id"] if "Id" in df.columns else None,  # Exclude ID column if present
      timeout_minutes=30,  # Adjust based on time constraints (default is 120)
      # Note: AutoML will test multiple algorithms including:
      # - sklearn (SVM, Logistic Regression, Decision Tree, Random Forest)
      # - LightGBM
      # - XGBoost (if available)
  )
  ```
- AutoML will:
  - Automatically split data into train/validation/test sets
  - Test multiple algorithms and hyperparameter combinations
  - Log all trials to MLflow
  - Generate trial notebooks (best trial gets a notebook)
  - Return an `AutoMLSummary` object

#### Step 3.3: Analyze AutoML Results

- Access the best trial: `best_trial = summary.best_trial`
- Display AutoML summary information:
  - `summary.trials` - list of all trials
  - `summary.best_trial.metrics` - metrics for best model
  - `summary.best_trial.params` - parameters for best model
  - `summary.best_trial.model_description` - description of best model
  - `summary.best_trial.evaluation_metric_score` - primary metric score
- Load the best model: `best_model = summary.best_trial.load_model()`
- Identify the model type/algorithm from `best_trial.model_description` or `best_trial.params`
- Store the best model type for use in Phase 4 (e.g., "RandomForest", "LightGBM", "SVM", etc.)
- Add markdown cell summarizing AutoML results and best model selection

### Phase 4: Hyperparameter Tuning with Hyperopt (Best Model Only)

**Important**: This phase only tunes the best model identified by AutoML in Phase 3. Do not tune all models - only the best one.

#### Step 4.1: Prepare Data for Hyperopt

- Since AutoML handled the data splitting, we need to prepare data for Hyperopt:
  - Load the original dataset: `df = spark.table("main.tomes_gen.iris")`
  - Convert to Pandas if needed: `iris_df = df.toPandas()`
  - Separate features and target:
    - Features: All columns except `Id` (if present) and `Species`
    - Target: `Species`
  - Encode target labels if needed (LabelEncoder if string labels)
  - Create train/test split:
    - Use `train_test_split()` with appropriate test size (e.g., 0.2 or 0.3)
    - Set `random_state` for reproducibility
    - Use `stratify=y` to maintain class distribution

#### Step 4.2: Define Objective Function

Create an objective function for Hyperopt that:

- Takes hyperparameter space as input
- Creates a new MLflow run within the function using `with mlflow.start_run(run_name="hyperopt_trial")`
- Instantiates the best model type (identified from AutoML Phase 3) with hyperparameters
- Trains the model: `model.fit(X_train, y_train)`
- Makes predictions: `y_pred = model.predict(X_test)`
- Evaluates on test set: `accuracy = accuracy_score(y_test, y_pred)`
- Logs metrics to MLflow: `mlflow.log_metric("accuracy", accuracy)`
- Returns dictionary with:
  - `loss`: negative accuracy (since Hyperopt minimizes)
  - `status`: `STATUS_OK`
  - Any additional metrics to track

#### Step 4.3: Define Hyperparameter Search Space

Define search spaces based on the best model type identified in Phase 3:

- **For SVM (SVC)**:
  - `C`: `hp.loguniform('C', -5, 5)`
  - `kernel`: `hp.choice('kernel', ['linear', 'rbf', 'poly'])`
  - `gamma`: `hp.loguniform('gamma', -5, 5)` (if kernel != 'linear')
- **For LogisticRegression**:
  - `C`: `hp.loguniform('C', -5, 5)`
  - `penalty`: `hp.choice('penalty', ['l1', 'l2', 'elasticnet'])`
  - `solver`: appropriate solver based on penalty (e.g., 'liblinear' for l1, 'lbfgs' for l2)
- **For DecisionTree**:
  - `max_depth`: `scope.int(hp.quniform('max_depth', 3, 20, 1))`
  - `min_samples_split`: `scope.int(hp.quniform('min_samples_split', 2, 20, 1))`
  - `min_samples_leaf`: `scope.int(hp.quniform('min_samples_leaf', 1, 10, 1))`
- **For RandomForest**:
  - `n_estimators`: `scope.int(hp.quniform('n_estimators', 50, 200, 10))`
  - `max_depth`: `scope.int(hp.quniform('max_depth', 3, 20, 1))`
  - `min_samples_split`: `scope.int(hp.quniform('min_samples_split', 2, 20, 1))`
- **For LightGBM (LGBMClassifier)**:
  - `n_estimators`: `scope.int(hp.quniform('n_estimators', 50, 300, 10))`
  - `max_depth`: `scope.int(hp.quniform('max_depth', 3, 15, 1))`
  - `learning_rate`: `hp.loguniform('learning_rate', -3, 0)`  # 0.001 to 1
  - `num_leaves`: `scope.int(hp.quniform('num_leaves', 20, 300, 10))`
  - `min_child_samples`: `scope.int(hp.quniform('min_child_samples', 5, 100, 5))`
  - `subsample`: `hp.uniform('subsample', 0.6, 1.0)`
  - `colsample_bytree`: `hp.uniform('colsample_bytree', 0.6, 1.0)`

#### Step 4.4: Configure SparkTrials for Parallel Execution

- Create `SparkTrials` object:
  ```python
  spark_trials = SparkTrials(parallelism=4)  # Adjust based on cluster capacity
  ```
- This enables parallel hyperparameter search across Spark executors

#### Step 4.5: Run Hyperopt Optimization

- Use `fmin()` to run optimization:
  ```python
  best_params = fmin(
      fn=objective,
      space=search_space,
      algo=tpe.suggest,
      max_evals=50,  # Adjust based on time constraints
      trials=spark_trials
  )
  ```
- Log the optimization process
- Display best parameters found
- Note: Each Hyperopt trial will create its own MLflow run, so all tuning attempts are tracked

#### Step 4.6: Train Final Model with Best Parameters

- Create final MLflow run with name like "best_tuned_model"
- Instantiate model with best hyperparameters from Hyperopt
- Train on full training set (or combine train+validation if using validation set)
- Evaluate on test set
- Log final model and metrics
- Compare performance with AutoML best model to show improvement from Hyperopt tuning

### Phase 5: Model Evaluation and Summary

#### Step 5.1: Final Model Evaluation

- Generate detailed evaluation metrics:
  - Confusion matrix
  - Classification report
  - Per-class metrics (precision, recall, F1-score)
- Visualize results (optional but recommended for demo):
  - Confusion matrix heatmap
  - Feature importance (if applicable to model type)

#### Step 5.2: MLflow Model Registration (Optional but Recommended)

- Register the best model in MLflow Model Registry:
  ```python
  mlflow.sklearn.log_model(
      model, 
      "model",
      registered_model_name="iris_classifier"
  )
  ```

#### Step 5.3: Summary and Documentation

- Add markdown cell summarizing:
  - Best model type and hyperparameters
  - Final accuracy and key metrics
  - Key learnings from the process
- Document the workflow for future reference

## Technical Requirements & Best Practices

### MLflow Specifics

- **Version**: Use MLflow 3.6.0 syntax and features
- **Autologging**: Ensure `mlflow.sklearn.autolog()` is called before any model training
- **Run Context**: All model training must be within `mlflow.start_run()` context
- **Parameter Logging**: Explicitly log key parameters even with autolog enabled
- **Experiment Tracking**: All runs should be in "iris_demo" experiment

### Code Organization

- Use clear cell structure:
  - Markdown cells for section headers
  - Code cells for implementation
  - Separate cells for each major step
- Add comments explaining key decisions
- Use descriptive variable names

### Error Handling

- Add try-except blocks for model training (in case of parameter incompatibilities)
- Handle cases where certain hyperparameter combinations are invalid
- Gracefully handle MLflow connection issues

### Documentation References

When implementing, consult:

- MLflow 3.6.0 documentation (use context7 tool if needed)
- Context7 mcp tool is available for you to use to reference current documentation and syntax as needed
- Databricks AutoML API reference: https://docs.databricks.com/aws/en/machine-learning/automl/automl-api-reference
- Databricks AutoML algorithms: https://docs.databricks.com/aws/en/machine-learning/automl/#automl-algorithms
- Scikit-learn classifier documentation
- LightGBM documentation (for LGBMClassifier)
- Hyperopt documentation (especially SparkTrials)
- Databricks MLflow quick start example: https://docs.databricks.com/aws/en/notebooks/source/mlflow/mlflow-quick-start-python-mlflow-3.html
- Hyperopt Spark ML example: https://assets.docs.databricks.com/_extras/notebooks/source/hyperopt-spark-ml.html

## Validation Checklist

Before considering the implementation complete, verify:

- [ ] All imports are correct and available (including `databricks.automl` and `lightgbm`)
- [ ] Data loads successfully from Unity Catalog
- [ ] MLflow tracking URI and experiment are set correctly
- [ ] Databricks AutoML runs successfully and completes
- [ ] AutoML tests multiple algorithms (sklearn, LightGBM, potentially XGBoost)
- [ ] AutoML best trial is identified and model is loaded
- [ ] Best model type/algorithm is correctly identified from AutoML results
- [ ] Hyperopt is only applied to the best model from AutoML (not all models)
- [ ] Hyperopt search space is defined for the correct model type (including LightGBM if applicable)
- [ ] SparkTrials is configured for parallel execution
- [ ] Hyperopt optimization completes successfully
- [ ] Final tuned model is trained and evaluated
- [ ] All AutoML and Hyperopt runs are visible in MLflow UI
- [ ] Model performance metrics are reasonable (accuracy > 0.9 expected for Iris dataset)
- [ ] Comparison is made between AutoML best model and Hyperopt-tuned model
- [ ] Notebook runs end-to-end without errors

## Questions for Clarification

1. **AutoML Experiment Name**: Should AutoML use the same experiment name ("iris_demo") as set in Phase 2, or should it create its own experiment? The plan allows for either approach.
2. **Model Registration**: Should the final model be registered in MLflow Model Registry, or just logged? I've included it as optional but recommended.
3. **AutoML Timeout**: What timeout should be used for AutoML? I've suggested 30 minutes, but this could be adjusted based on time constraints (default is 120 minutes).
4. **Hyperopt Trials**: How many trials should be run? I've suggested 50, but this could be adjusted based on time constraints and cluster capacity.
5. **Data Splitting**: Should we use AutoML's automatic data splitting, or manually split data before AutoML? The plan uses AutoML's automatic splitting, but manual splitting could provide more control.

## Notes for Implementation

- The Iris dataset is small (150 samples), so training will be fast
- All models should achieve high accuracy (>90%) on this dataset
- The focus is on demonstrating the MLflow, Databricks AutoML, and Hyperopt workflow, not necessarily achieving the absolute best performance
- Databricks AutoML will automatically test multiple algorithms including:
  - sklearn models (SVM, Logistic Regression, Decision Tree, Random Forest)
  - LightGBM
  - XGBoost (if available in the environment)
- AutoML handles data preprocessing, feature engineering, and model selection automatically
- Hyperopt is used only to further tune the best model identified by AutoML
- Ensure all code follows Python best practices and is well-commented
- Test the notebook execution in the actual Databricks Connect environment before finalizing
- Note: AutoML may take longer than manual model training, but it provides a comprehensive comparison of multiple algorithms
