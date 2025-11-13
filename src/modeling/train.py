from pathlib import Path
import typer
from loguru import logger
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from scipy.stats import uniform, randint

from src.config import MODELS_DIR, PROCESSED_DATA_PATH, TEST_PERCENTAGE, SPLIT_RANDOM_STATE, NUM_CROSSVAL_FOLDS, \
    XGB_RANDOM_STATE, XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, \
    NN_RANDOM_STATE, NN_HIDDEN_LAYER_SIZES, NN_ACTIVATION, NN_SOLVER, NN_LEARNING_RATE_INIT, NN_MAX_ITER, NN_EARLY_STOPPING, NN_N_ITER_NO_CHANGE


def train_and_eval(model, model_metric_dict, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    model_metric_dict['train_mse'].append(train_mse)
    model_metric_dict['test_mse'].append(test_mse)
    model_metric_dict['train_mae'].append(train_mae)
    model_metric_dict['test_mae'].append(test_mae)
    model_metric_dict['train_r2'].append(train_r2)
    model_metric_dict['test_r2'].append(test_r2)


def compute_mean_var(metrics_dict):
    return {key: {'mean': np.mean(values), 'var': np.var(values)} for key, values in metrics_dict.items()}


def single_experiment(X, y):
    linear_metrics = {'train_mse': [], 'test_mse': [], 'train_mae': [], 'test_mae': [], 'train_r2': [], 'test_r2': []}
    xgb_metrics    = {'train_mse': [], 'test_mse': [], 'train_mae': [], 'test_mae': [], 'train_r2': [], 'test_r2': []}
    nn_metrics     = {'train_mse': [], 'test_mse': [], 'train_mae': [], 'test_mae': [], 'train_r2': [], 'test_r2': []}

    for crossval_fold in range(NUM_CROSSVAL_FOLDS):
        fold_seed = SPLIT_RANDOM_STATE + crossval_fold
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_PERCENTAGE, random_state=fold_seed)

        # Linear model (baseline)
        linear_model = LinearRegression()
        train_and_eval(linear_model, linear_metrics, X_train, y_train, X_test, y_test)

        # XGBoost model
        xgb_model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE,
                random_state=XGB_RANDOM_STATE + crossval_fold,
                verbosity=0
            )
        )
        train_and_eval(xgb_model, xgb_metrics, X_train, y_train, X_test, y_test)

        # NN Model
        nn_model = MLPRegressor(
            hidden_layer_sizes=NN_HIDDEN_LAYER_SIZES,
            activation=NN_ACTIVATION,
            solver=NN_SOLVER,
            learning_rate_init=NN_LEARNING_RATE_INIT,
            max_iter=NN_MAX_ITER, # Max number of epochs
            random_state=NN_RANDOM_STATE + crossval_fold,
            early_stopping=NN_EARLY_STOPPING,
            n_iter_no_change=NN_N_ITER_NO_CHANGE,
            verbose=False
        )
        train_and_eval(nn_model, nn_metrics, X_train, y_train, X_test, y_test)
    
    # Compute mean and variance
    linear_summary = compute_mean_var(linear_metrics)
    xgb_summary = compute_mean_var(xgb_metrics)
    nn_summary = compute_mean_var(nn_metrics)

    # Convert to DataFrame
    all_summaries = {
        'Linear Regression': linear_summary,
        'XGBoost': xgb_summary,
        'Neural Network': nn_summary
    }

    rows = []
    for model_name, metrics in all_summaries.items():
        row = {'model': model_name}
        for metric_name, stat in metrics.items():
            row[f"{metric_name}_mean"] = stat['mean']
            row[f"{metric_name}_var"] = stat['var']
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


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