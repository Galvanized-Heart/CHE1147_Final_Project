from pathlib import Path
import copy
import typer
from loguru import logger
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Consider pearson or ther correl metrics
from xgboost import XGBRegressor
from scipy.stats import uniform, randint

from src.config import *



def train_and_eval(model, model_metric_dict, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    # Save metrics
    model_metric_dict['train_mse'].append(train_mse)
    model_metric_dict['val_mse'].append(val_mse)
    model_metric_dict['train_mae'].append(train_mae)
    model_metric_dict['val_mae'].append(val_mae)
    model_metric_dict['train_r2'].append(train_r2)
    model_metric_dict['val_r2'].append(val_r2)


# Depracated?
def compute_mean_var(metrics_dict):
    return {key: {'mean': np.mean(values), 'var': np.var(values)} for key, values in metrics_dict.items()}



def single_experiment(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    model_params: dict = None
):    
    if model_params is None:
        model_params = {}

    X_train_subset = X_train[feature_cols]
    y_train_subset = y_train[target_cols]
    X_val_subset = X_val[feature_cols]
    y_val_subset = y_val[target_cols]

    linear_metrics = {'train_mse': [], 'val_mse': [], 'train_mae': [], 'val_mae': [], 'train_r2': [], 'val_r2': []}
    xgb_metrics    = copy.deepcopy(linear_metrics)
    nn_metrics     = copy.deepcopy(linear_metrics)
    
    # Linear model
    linear_model = LinearRegression()
    train_and_eval(linear_model, linear_metrics, X_train_subset, y_train_subset, X_val=X_val_subset, y_val=y_val_subset)

    # XGBoost model
    xgb_params = model_params.get('XGBoost', {})
    final_xgb_params = {
        'n_estimators': XGB_N_ESTIMATORS,
        'max_depth': XGB_MAX_DEPTH,
        'learning_rate': XGB_LEARNING_RATE,
        'random_state': XGB_RANDOM_STATE,
        'verbosity': 0,
        **xgb_params # Overwrite defaults
    }
    xgb_model = MultiOutputRegressor(XGBRegressor(**final_xgb_params))
    train_and_eval(xgb_model, xgb_metrics, X_train_subset, y_train_subset, X_val=X_val_subset, y_val=y_val_subset)

    # NN Model
    nn_params = model_params.get('Neural Network', {})
    final_nn_params = {
        'hidden_layer_sizes': NN_HIDDEN_LAYER_SIZES,
        'activation': NN_ACTIVATION,
        'solver': NN_SOLVER,
        'learning_rate_init': NN_LEARNING_RATE_INIT,
        'max_iter': NN_MAX_ITER,
        'random_state': NN_RANDOM_STATE,
        'early_stopping': NN_EARLY_STOPPING,
        'n_iter_no_change': NN_N_ITER_NO_CHANGE,
        'verbose': 0,
        **nn_params # Overwrite defaults
    }
    nn_model = MLPRegressor(**final_nn_params)
    train_and_eval(nn_model, nn_metrics, X_train_subset, y_train_subset, X_val=X_val_subset, y_val=y_val_subset)

    results = [
        {'model': 'Linear Regression', **linear_metrics},
        {'model': 'XGBoost', **xgb_metrics},
        {'model': 'Neural Network', **nn_metrics},
    ]
    
    # The metrics dicts have lists with one item, so we extract it
    for result in results:
        for key, val in result.items():
            if isinstance(val, list):
                result[key] = val[0]

    return pd.DataFrame(results)



def run_kfold_validation(
    splits_dict: dict, 
    best_params: dict,
    feature_cols: list[str],
    target_cols: list[str],
    experiment_name: str
):
    logger.info(f"Starting K-Fold Cross-Validation for {experiment_name}")
    all_fold_results = []
    
    # Loop through each fold defined in the splits dictionary
    for fold_name, paths in splits_dict['folds'].items():
        logger.info(f"Running validation for {fold_name}...")
        
        # Unpack paths for this fold
        X_train_path, y_train_path = paths['train']
        X_val_path, y_val_path = paths['val']
        
        # Load data from parquet files
        X_train = pd.read_parquet(X_train_path)
        y_train = pd.read_parquet(y_train_path)
        X_val = pd.read_parquet(X_val_path)
        y_val = pd.read_parquet(y_val_path)
        
        # Run all models on this specific fold
        fold_results_df = single_experiment(
            X_train, y_train, X_val, y_val,
            feature_cols=feature_cols,
            target_cols=target_cols,
            model_params=best_params
        )
        fold_results_df['fold'] = fold_name
        all_fold_results.append(fold_results_df)

    # Combine results from all folds into a single DataFrame
    final_results_df = pd.concat(all_fold_results, ignore_index=True)
    
    logger.info("K-Fold Cross-Validation Complete")
    print(f"\nFull Results for {experiment_name} Across All Folds:")
    print(final_results_df.round(4))

    # Calculate and display the mean and standard deviation of metrics across folds
    summary = final_results_df.drop(columns=['fold']).groupby('model').agg(['mean', 'std'])
    
    logger.success(f"Final Performance Summary for {experiment_name} (Mean +/- Std Dev)")
    print("\nAggregated Performance:")
    print(summary['val_r2'].round(4))
    
    # Save summary DataFrame
    summary_path = CV_RESULTS_DIR / 'summary.csv'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path)
    logger.success(f"Saved results for {experiment_name} to {summary_path}")

    return summary


def run_all_experiments(splits_dict: dict, best_params: dict):
    all_targets = KCAT_TARGET_COLS + KM_TARGET_COLS

    experiments = {
        "basic_features": {
            "features": BASIC_FEATS_COLS, 
            "targets": all_targets,
        },
        "basic_plus_conditions": {
            "features": BASIC_FEATS_COLS + PH_FEATS_COLS + TEMPERATURE_FEATS_COLS,
            "targets": all_targets,
        },
        "advanced_features": {
            "features": ADVANCED_FEATS_COLS, 
            "targets": all_targets,
        },
        "advanced_plus_conditions": {
            "features": ADVANCED_FEATS_COLS + PH_FEATS_COLS + TEMPERATURE_FEATS_COLS,
            "targets": all_targets,
        },
        "all_features": {
            "features": BASIC_FEATS_COLS + ADVANCED_FEATS_COLS + PH_FEATS_COLS + TEMPERATURE_FEATS_COLS,
            "targets": all_targets,
        }
    }

    for name, config in experiments.items():
        run_kfold_validation(
            splits_dict=splits_dict,
            best_params=best_params,
            feature_cols=config["features"],
            target_cols=config["targets"],
            experiment_name=name
        )


# Depracated?
def train():
    """
    Loads processed training and validation data, trains baseline and XGBoost models
    for kcat and KM prediction, performs hyperparameter tuning, and saves the best models.
    """
    logger.info("--- Loading Processed Data ---")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    X = df.drop(columns=['log_kcat', 'log_km'])
    y = df[['log_kcat', 'log_km']]

    

if __name__ == "__main__":
    app()