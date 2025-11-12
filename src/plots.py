import typer
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from loguru import logger

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, MODELS_DIR

# Set a consistent, publication-quality style for plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

app = typer.Typer()


def generate_parity_plot(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str,
    output_path: Path,
):
    """
    Generates and saves a parity plot (predicted vs. actual values).

    Args:
        y_true: Series of true target values.
        y_pred: Series of predicted target values.
        title: The title for the plot.
        output_path: The path to save the plot image.
    """
    plt.figure(figsize=(6, 6))
    
    ax = sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor='k', s=50)
    
    # Determine the limits for the 45-degree line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),  # min of both axes
        max(ax.get_xlim()[1], ax.get_ylim()[1]),  # max of both axes
    ]
    
    # Plot the 45-degree line for reference
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel(f"Actual {y_true.name}", fontsize=12)
    ax.set_ylabel(f"Predicted {y_true.name}", fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved parity plot to {output_path}")


def generate_feature_importance_plot(
    model,
    feature_names: list,
    title: str,
    output_path: Path,
    top_n: int = 20,
):
    """
    Generates and saves a feature importance plot from a tree-based model.

    Args:
        model: A trained model with a `feature_importances_` attribute.
        feature_names: A list of names for the features.
        title: The title for the plot.
        output_path: The path to save the plot image.
        top_n: The number of top features to display.
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning(f"Model of type {type(model)} does not have feature_importances_. Skipping plot.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_importances = importances.nlargest(top_n).sort_values(ascending=True)

    plt.figure(figsize=(10, 8))
    
    ax = top_importances.plot(kind='barh', color=sns.color_palette('viridis', n_colors=top_n))
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    
    # Add labels to the bars
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', fontsize=8, padding=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importance plot to {output_path}")


@app.command()
def main(
    predictions_path: Path = typer.Option(PROCESSED_DATA_DIR / "test_predictions.csv", help="Path to test predictions CSV file."),
    features_path: Path = typer.Option(PROCESSED_DATA_DIR / "X_train.parquet", help="Path to training features to get column names."),
    models_dir: Path = typer.Option(MODELS_DIR, help="Directory where trained models are saved."),
    output_dir: Path = typer.Option(FIGURES_DIR, help="Directory to save the plots."),
):
    """
    Generates all result visualizations (parity plots, feature importances)
    for the final report.
    """
    logger.info("--- Starting Plot Generation ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading predictions from {predictions_path}")
    predictions_df = pd.read_csv(predictions_path)
    
    logger.info(f"Loading feature names from {features_path}")
    feature_names = pd.read_parquet(features_path).columns.tolist()

    targets = ['log_kcat', 'log_km']
    
    for target in targets:
        logger.info(f"--- Generating plots for target: {target} ---")
        
        # --- Parity Plot ---
        y_true = predictions_df[target]
        y_pred = predictions_df[f'xgboost_{target}_pred']
        
        generate_parity_plot(
            y_true=y_true,
            y_pred=y_pred,
            title=f"XGBoost Performance for {target.replace('_', ' ').title()}",
            output_path=output_dir / f"parity_plot_xgboost_{target}.png"
        )
        
        # --- Feature Importance Plot ---
        try:
            model_path = models_dir / f'xgboost_model_{target}.joblib'
            model = joblib.load(model_path)
            
            generate_feature_importance_plot(
                model=model,
                feature_names=feature_names,
                title=f"Top 20 Features for {target.replace('_', ' ').title()}",
                output_path=output_dir / f"feature_importance_{target}.png"
            )
        except FileNotFoundError:
            logger.error(f"Model file not found at {model_path}. Cannot generate feature importance plot.")
        except Exception as e:
            logger.error(f"An error occurred while generating feature importance plot for {target}: {e}")

    logger.success("--- Plot generation complete! Figures saved in reports/figures/ ---")


if __name__ == "__main__":
    app()