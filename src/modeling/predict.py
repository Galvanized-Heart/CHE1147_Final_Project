from pathlib import Path
import typer
from loguru import logger
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def evaluate_model(y_true, y_pred, model_name, target_name):
    """Calculates and logs regression metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    logger.info(f"--- {model_name} on {target_name} (Test Set) ---")
    logger.info(f"R-squared: {r2:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    return {"Model": model_name, "Target": target_name, "R2": r2, "MAE": mae, "RMSE": rmse}

@app.command()
def main(
    processed_data_dir: Path = typer.Option(PROCESSED_DATA_DIR, help="Path to processed data."),
    models_dir: Path = typer.Option(MODELS_DIR, help="Path to saved models."),
    predictions_path: Path = typer.Option(PROCESSED_DATA_DIR / "test_predictions.csv", help="Path to save test predictions."),
):
    """
    Loads trained models and test data, performs inference, evaluates performance,
    and saves the predictions.
    """
    logger.info("--- Evaluating Models on the Test Set ---")
    X_test = pd.read_parquet(processed_data_dir / 'X_test.parquet')
    y_test = pd.read_parquet(processed_data_dir / 'y_test.parquet')

    targets = ['log_kcat', 'log_km']
    models_to_eval = ['dummy', 'xgboost']
    results = []
    
    # Create a dataframe to store predictions for plotting
    predictions_df = y_test.copy()

    for target in targets:
        y_true = y_test[target]
        for model_name in models_to_eval:
            model_path = models_dir / f'{model_name}_model_{target}.joblib'
            logger.info(f"Loading model: {model_path}")
            model = joblib.load(model_path)
            
            y_pred = model.predict(X_test)
            predictions_df[f'{model_name}_{target}_pred'] = y_pred
            
            # Evaluate and store results
            metrics = evaluate_model(y_true, y_pred, model_name.capitalize(), target)
            results.append(metrics)

    # Save predictions
    predictions_df.to_csv(predictions_path)
    logger.success(f"Saved predictions with true values to {predictions_path}")
    
    # Display final results table
    results_df = pd.DataFrame(results)
    logger.info("\n--- Final Performance Summary on Test Set ---")
    print(results_df.to_markdown(index=False))

if __name__ == "__main__":
    app()