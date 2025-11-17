from loguru import logger
import pandas as pd
import joblib
from scipy.stats import uniform, randint, loguniform
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

from config import HPO_RESULTS_DIR, HPO_ROUNDS, SPLIT_RANDOM_STATE, ADVANCED_FEATS_COLS, BASIC_FEATS_COLS, \
PH_FEATS_COLS, TEMPERATURE_FEATS_COLS, KCAT_TARGET_COLS, KM_TARGET_COLS


def run_hpo(
    splits_dict: dict,
    feature_cols: list[str],
    experiment_name: str
) -> dict:
    logger.info(f"Starting Hyperparameter Optimization using Pre-defined Split for {experiment_name}")
    logger.info(f"Using {len(feature_cols)} features.")

    # Read data
    X_train_path, y_train_path = splits_dict['hpo']['train']
    X_val_path, y_val_path = splits_dict['hpo']['val']
    X_hpo_train = pd.read_parquet(X_train_path)
    y_hpo_train = pd.read_parquet(y_train_path)
    X_hpo_val = pd.read_parquet(X_val_path)
    y_hpo_val = pd.read_parquet(y_val_path)

    # Select feature columns
    X_hpo_train = X_hpo_train[feature_cols]
    X_hpo_val = X_hpo_val[feature_cols]

    # Combine for RandomizedSearchCV
    X_combined = pd.concat([X_hpo_train, X_hpo_val], ignore_index=True)
    y_combined = pd.concat([y_hpo_train, y_hpo_val], ignore_index=True)

    # Create PredefinedSplit index so best_score_ uses validation metrics
    split_index = [-1] * len(X_hpo_train) + [0] * len(X_hpo_val)
    ps = PredefinedSplit(test_fold=split_index)

    # HPO configs
    hpo_configs = [
        {
            'name': 'XGBoost',
            'estimator': MultiOutputRegressor(XGBRegressor(random_state=SPLIT_RANDOM_STATE, verbosity=0)),
            'params': {
                'estimator__n_estimators': randint(100, 1000),
                'estimator__max_depth': randint(3, 15),
                'estimator__learning_rate': loguniform(0.01, 0.3),
                'estimator__subsample': uniform(0.6, 0.4),
                'estimator__colsample_bytree': uniform(0.6, 0.4)
            }
        },
        {
            'name': 'Neural Network',
            'estimator': MLPRegressor(random_state=SPLIT_RANDOM_STATE, early_stopping=True, n_iter_no_change=10, max_iter=500),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam'],
                'alpha': loguniform(1e-5, 1e-2),
                'learning_rate_init': loguniform(1e-4, 1e-2),
            }
        }
    ]
    
    all_best_params = {}

    for config in hpo_configs:
        logger.info(f"Tuning {config['name']} for {experiment_name} with {HPO_ROUNDS} rounds")

        # Initialize HPO object
        random_search = RandomizedSearchCV(
            config['estimator'],
            param_distributions=config['params'],
            n_iter=2,
            cv=ps,
            scoring='r2',
            n_jobs=1,
            random_state=SPLIT_RANDOM_STATE,
            verbose=3
        )

        # Perform hyperparam search
        random_search.fit(X_combined, y_combined)
        
        # Report scores
        logger.success(f"Best validation R2 score for {config['name']}: {random_search.best_score_:.4f}")
        
        best_params_cleaned = {k.split('__')[-1]: v for k, v in random_search.best_params_.items()}
        all_best_params[config['name']] = best_params_cleaned
        
        print("Best hyperparameters found:")
        print(best_params_cleaned)

        # Save params
        hpo_results_path = HPO_RESULTS_DIR / f'best_params_{experiment_name}.joblib'
        hpo_results_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(all_best_params, hpo_results_path)
        logger.success(f"Saved best parameters for {experiment_name} to {hpo_results_path}")

    return all_best_params

def run_all_hpo(splits_dict: dict):
    """
    Orchestrates HPO runs for different predefined feature sets.
    """
    hpo_experiments = {
        "basic_plus_conditions": {
            "features": BASIC_FEATS_COLS + PH_FEATS_COLS + TEMPERATURE_FEATS_COLS,
        },
        "all_features": {
            "features": BASIC_FEATS_COLS + ADVANCED_FEATS_COLS + PH_FEATS_COLS + TEMPERATURE_FEATS_COLS,
        }
    }

    for name, config in hpo_experiments.items():
        run_hpo(
            splits_dict=splits_dict,
            feature_cols=config["features"],
            experiment_name=name
        )