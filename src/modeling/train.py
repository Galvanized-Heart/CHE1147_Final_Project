from pathlib import Path
import typer
from loguru import logger
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    processed_data_dir: Path = typer.Option(PROCESSED_DATA_DIR, help="Path to processed data directory."),
    models_output_dir: Path = typer.Option(MODELS_DIR, help="Directory to save trained models."),
):
    """
    Loads processed training and validation data, trains baseline and XGBoost models
    for kcat and KM prediction, performs hyperparameter tuning, and saves the best models.
    """
    logger.info("--- Loading Processed Data ---")
    X_train = pd.read_parquet(processed_data_dir / 'X_train.parquet')
    y_train = pd.read_parquet(processed_data_dir / 'y_train.parquet')
    X_val = pd.read_parquet(processed_data_dir / 'X_val.parquet')
    y_val = pd.read_parquet(processed_data_dir / 'y_val.parquet')

    targets = ['log_kcat', 'log_km']
    models_output_dir.mkdir(parents=True, exist_ok=True)
    
    for target in targets:
        logger.info(f"\n--- Training models for {target} ---")
        y_train_target = y_train[target]
        y_val_target = y_val[target]
        
        # 1. Baseline Model
        logger.info("Training Baseline (DummyRegressor)...")
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train_target)
        model_path = models_output_dir / f'dummy_model_{target}.joblib'
        joblib.dump(dummy, model_path)
        logger.success(f"Saved baseline model to {model_path}")

        # 2. Advanced Model (XGBoost) with RandomizedSearchCV
        logger.info("Training Advanced Model (XGBoost) with Randomized Search...")
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.2),
            'subsample': uniform(0.6, 0.4), # range is 0.6 to 1.0
            'colsample_bytree': uniform(0.6, 0.4)
        }
        xgbr = xgb.XGBRegressor(random_state=42, early_stopping_rounds=10, n_jobs=-1)
        
        random_search = RandomizedSearchCV(
            estimator=xgbr, param_distributions=param_dist, n_iter=20, # 20 iterations for speed
            cv=3, verbose=2, random_state=42, scoring='r2'
        )
        
        random_search.fit(X_train, y_train_target, eval_set=[(X_val, y_val_target)], verbose=False)

        best_model = random_search.best_estimator_
        logger.info(f"Best parameters for {target}: {random_search.best_params_}")
        logger.info(f"Best validation R2 score for {target}: {random_search.best_score_:.4f}")

        model_path = models_output_dir / f'xgboost_model_{target}.joblib'
        joblib.dump(best_model, model_path)
        logger.success(f"Saved best XGBoost model to {model_path}")

    logger.success("Model training complete for all targets.")

if __name__ == "__main__":
    app()