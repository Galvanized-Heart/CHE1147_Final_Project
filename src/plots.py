# import typer
# import joblib
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from pathlib import Path
# from loguru import logger

# from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, MODELS_DIR

# # Set a consistent, publication-quality style for plots
# sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# app = typer.Typer()

# def generate_parity_plot(
#     y_true: pd.Series,
#     y_pred: pd.Series,
#     title: str,
#     output_path: Path,
# ):
#     """
#     Generates and saves a parity plot (predicted vs. actual values).

#     Args:
#         y_true: Series of true target values.
#         y_pred: Series of predicted target values.
#         title: The title for the plot.
#         output_path: The path to save the plot image.
#     """
#     plt.figure(figsize=(6, 6))
    
#     ax = sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor='k', s=50)
    
#     # Determine the limits for the 45-degree line
#     lims = [
#         min(ax.get_xlim()[0], ax.get_ylim()[0]),  # min of both axes
#         max(ax.get_xlim()[1], ax.get_ylim()[1]),  # max of both axes
#     ]
    
#     # Plot the 45-degree line for reference
#     ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
    
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_xlim(lims)
#     ax.set_ylim(lims)
    
#     ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
#     ax.set_xlabel(f"Actual {y_true.name}", fontsize=12)
#     ax.set_ylabel(f"Predicted {y_true.name}", fontsize=12)
#     ax.legend()
    
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()
#     logger.info(f"Saved parity plot to {output_path}")


# def generate_feature_importance_plot(
#     model,
#     feature_names: list,
#     title: str,
#     output_path: Path,
#     top_n: int = 20,
# ):
#     """
#     Generates and saves a feature importance plot from a tree-based model.

#     Args:
#         model: A trained model with a `feature_importances_` attribute.
#         feature_names: A list of names for the features.
#         title: The title for the plot.
#         output_path: The path to save the plot image.
#         top_n: The number of top features to display.
#     """
#     if not hasattr(model, 'feature_importances_'):
#         logger.warning(f"Model of type {type(model)} does not have feature_importances_. Skipping plot.")
#         return

#     importances = pd.Series(model.feature_importances_, index=feature_names)
#     top_importances = importances.nlargest(top_n).sort_values(ascending=True)

#     plt.figure(figsize=(10, 8))
    
#     ax = top_importances.plot(kind='barh', color=sns.color_palette('viridis', n_colors=top_n))
    
#     ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
#     ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
#     ax.set_ylabel("Features", fontsize=12)
    
#     # Add labels to the bars
#     for i in ax.containers:
#         ax.bar_label(i, fmt='%.3f', fontsize=8, padding=3)

#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()
#     logger.info(f"Saved feature importance plot to {output_path}")


# @app.command()
# def main(
#     predictions_path: Path = typer.Option(PROCESSED_DATA_DIR / "test_predictions.csv", help="Path to test predictions CSV file."),
#     features_path: Path = typer.Option(PROCESSED_DATA_DIR / "X_train.parquet", help="Path to training features to get column names."),
#     models_dir: Path = typer.Option(MODELS_DIR, help="Directory where trained models are saved."),
#     output_dir: Path = typer.Option(FIGURES_DIR, help="Directory to save the plots."),
# ):
#     """
#     Generates all result visualizations (parity plots, feature importances)
#     for the final report.
#     """
#     logger.info("--- Starting Plot Generation ---")
#     output_dir.mkdir(parents=True, exist_ok=True)

#     logger.info(f"Loading predictions from {predictions_path}")
#     predictions_df = pd.read_csv(predictions_path)
    
#     logger.info(f"Loading feature names from {features_path}")
#     feature_names = pd.read_parquet(features_path).columns.tolist()

#     targets = ['log_kcat', 'log_km']
    
#     for target in targets:
#         logger.info(f"--- Generating plots for target: {target} ---")
        
#         # --- Parity Plot ---
#         y_true = predictions_df[target]
#         y_pred = predictions_df[f'xgboost_{target}_pred']
        
#         generate_parity_plot(
#             y_true=y_true,
#             y_pred=y_pred,
#             title=f"XGBoost Performance for {target.replace('_', ' ').title()}",
#             output_path=output_dir / f"parity_plot_xgboost_{target}.png"
#         )
        
#         # --- Feature Importance Plot ---
#         try:
#             model_path = models_dir / f'xgboost_model_{target}.joblib'
#             model = joblib.load(model_path)
            
#             generate_feature_importance_plot(
#                 model=model,
#                 feature_names=feature_names,
#                 title=f"Top 20 Features for {target.replace('_', ' ').title()}",
#                 output_path=output_dir / f"feature_importance_{target}.png"
#             )
#         except FileNotFoundError:
#             logger.error(f"Model file not found at {model_path}. Cannot generate feature importance plot.")
#         except Exception as e:
#             logger.error(f"An error occurred while generating feature importance plot for {target}: {e}")

#     logger.success("--- Plot generation complete! Figures saved in reports/figures/ ---")


# if __name__ == "__main__":
#     app()

import typer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

# Import necessary metrics from scikit-learn
try:
    from sklearn.metrics import r2_score, mean_squared_error
except ImportError:
    # This will be handled in the main() command
    pass

# Initialize Typer application
app = typer.Typer(help="Generates a parity plot and an optional R-squared bar graph from CSV files.")

# Set a consistent, publication-quality style for plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def generate_plot(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    output_path: Path,
    title: str,
):
    
    #Generates and saves a parity plot (Predicted vs. Actual).
    
    # Extract the series for convenience and readability
    y_true = df[actual_col]
    y_pred = df[predicted_col]

    # Calculate R^2 and RMSE
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse**0.5 # Calculate the square root of MSE

    # Create the figure
    plt.figure(figsize=(7, 7))

    # Generate the scatter plot
    ax = sns.scatterplot(
        x=y_true, 
        y=y_pred, 
        alpha=0.6, 
        edgecolor='k', 
        s=60,
        label=f"Data Points\n$R^2$: {r2:.3f}\nRMSE: {rmse:.3f}"
    )

    # --- Setup the 45-degree line for reference ---
    
    # 1. Determine the limits based on the data range
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    
    # Add a small buffer to the limits
    buffer = (max_val - min_val) * 0.05
    lims = [min_val - buffer, max_val + buffer]

    # 2. Plot the 45-degree line (Perfect Prediction)
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction ($y=x$)', zorder=0)

    # --- Final Plot Styling ---
    
    # Set axis limits and aspect ratio to ensure a square plot
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Set title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(f"Actual Values ({actual_col})", fontsize=14)
    ax.set_ylabel(f"Predicted Values ({predicted_col})", fontsize=14)
    ax.legend(loc='upper left', frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Successfully saved parity plot to {output_path.resolve()}")


def generate_bar_plot(df: pd.DataFrame, model_col: str, r2_col: str, output_path: Path):
   
    #Generates and saves a bar plot of R-squared values by model type.
    #Flipped to a VERTICAL bar chart.
    
    # Sort the dataframe by R-squared value in descending order
    df_sorted = df.sort_values(by=r2_col, ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 6)) # Adjusted figsize for better vertical display
    
    # Flipped axis: x=Model Type (categorical), y=R^2 Value (quantitative)
    ax = sns.barplot(
        x=model_col,  # Model names on the x-axis
        y=r2_col,     # R^2 values on the y-axis
        data=df_sorted, 
        palette="viridis",  
        edgecolor=".2"
    )
    
    # Add R^2 values as labels on top of the bars
    for index, row in df_sorted.iterrows():
        # Text label is added on top of the bar (y=R2 value)
        ax.text(
            index,                         # x position (index of the bar, 0, 1, 2...)
            row[r2_col] + 0.01,            # y position (R2 value + slight offset above the bar)
            f'{row[r2_col]:.4f}',          # text to display
            color='black', 
            ha="center",                   # center the text horizontally over the bar
            va="bottom",                   # align the text to the bottom (just above the bar)
            fontsize=10
        )

    ax.set_title("Model Performance Comparison ($R^2$)", fontsize=16, fontweight='bold', pad=15)
    # Swapped axis labels to match the vertical plot
    ax.set_xlabel("Model Type", fontsize=14) 
    ax.set_ylabel("$R^2$ Value", fontsize=14) 
    
    # Set y-axis limit slightly past the maximum R2 value to accommodate labels
    max_r2 = df_sorted[r2_col].max()
    # Use set_ylim for vertical plot
    ax.set_ylim(0, max_r2 * 1.1) 
    
    # Optional: Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Successfully saved metrics bar plot to {output_path.resolve()}")


@app.command()
def main(
    # --- Parity Plot Arguments ---
    input_file: Path = typer.Argument(
        ..., 
        exists=True, 
        file_okay=True, 
        dir_okay=False, 
        help="Path to the input CSV file for the parity plot (Actual/Predicted)."
    ),
    actual_col: str = typer.Option(
        "Actual", 
        "--actual-col", 
        "-a", 
        help="Column name for the actual (true) values in the parity file."
    ),
    predicted_col: str = typer.Option(
        "Predicted", 
        "--predicted-col", 
        "-p", 
        help="Column name for the predicted values in the parity file."
    ),
    output_file: Path = typer.Option(
        "parity_plot.png", 
        "--output-file", 
        "-o", 
        help="Path to save the output PNG file for the parity plot."
    ),
    plot_title: str = typer.Option(
        "Model Parity Check", 
        "--title", 
        "-t", 
        help="Title for the generated parity plot."
    ),
    
    # --- Bar Graph Arguments ---
    metrics_file: Optional[Path] = typer.Option(
        None,
        "--metrics-file",
        "-m",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Optional path to a second CSV containing model metrics (e.g., 'Model', 'R_squared')."
    ),
    metrics_output_file: Path = typer.Option(
        "metrics_bar_plot.png",
        "--metrics-output-file",
        "-mo",
        help="Path to save the output PNG file for the bar plot."
    ),
    model_col: str = typer.Option(
        "Model",
        "--model-col",
        help="Column name for the model type in the metrics file."
    ),
    r2_col: str = typer.Option(
        "R_squared",
        "--r2-col",
        help="Column name for the R-squared value in the metrics file."
    ),
):
    """
    Loads CSV data(s), generates the plot(s), and saves them.
    """
    try:
        # Check for scikit-learn dependency
        try:
            import sklearn.metrics
        except ImportError:
            typer.echo("Error: scikit-learn is required. Please install with 'pip install scikit-learn'.", err=True)
            raise typer.Exit(code=1)

        # 1. Generate Parity Plot
        print(f"Loading parity data from: {input_file.resolve()}")
        df_parity = pd.read_csv(input_file)

        if actual_col not in df_parity.columns:
            typer.echo(f"Error: Actual column '{actual_col}' not found in the parity file.", err=True)
            raise typer.Exit(code=1)
        if predicted_col not in df_parity.columns:
            typer.echo(f"Error: Predicted column '{predicted_col}' not found in the parity file.", err=True)
            raise typer.Exit(code=1)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        generate_plot(
            df=df_parity,
            actual_col=actual_col,
            predicted_col=predicted_col,
            output_path=output_file,
            title=plot_title
        )

        # 2. Generate Bar Graph (if metrics file is provided)
        if metrics_file:
            print(f"\nLoading metrics data from: {metrics_file.resolve()}")
            df_metrics = pd.read_csv(metrics_file)
            
            if model_col not in df_metrics.columns or r2_col not in df_metrics.columns:
                typer.echo(f"Error: Metrics file must contain '{model_col}' and '{r2_col}' columns for the bar plot.", err=True)
                raise typer.Exit(code=1)

            metrics_output_file.parent.mkdir(parents=True, exist_ok=True)
            generate_bar_plot(
                df=df_metrics,
                model_col=model_col,
                r2_col=r2_col,
                output_path=metrics_output_file
            )

    except Exception as e:
        typer.echo(f"An unexpected error occurred: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()