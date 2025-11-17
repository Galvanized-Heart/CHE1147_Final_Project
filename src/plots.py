import typer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List

# --- HARDCODED FILE PATH ---
# The path to the Parquet file is now fixed in the code.
PARQUET_FILE_PATH = Path("")
# ---------------------------

# Note: pandas requires 'pyarrow' or 'fastparquet' to read Parquet files.
# Install them with: pip install pyarrow fastparquet

# Import necessary metrics from scikit-learn
try:
    from sklearn.metrics import r2_score, mean_squared_error
except ImportError:
    # This will be handled in the main() command
    pass

# Initialize Typer application
app = typer.Typer(help="Generates a parity plot and an optional R-squared bar graph from a single hardcoded Parquet file.")

# Set a consistent, publication-quality style for plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def generate_plot(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    output_path: Path,
    title: str,
):
    """Generates and saves a parity plot (Predicted vs. Actual)."""
    
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
    """Generates and saves a bar plot of R-squared values by model type."""
    
    # Sort the dataframe by R-squared value in descending order
    df_sorted = df.sort_values(by=r2_col, ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 6)) # Adjusted figsize for better vertical display
    
    # Bar chart: x=Model Type (categorical), y=R^2 Value (quantitative)
    ax = sns.barplot(
        x=model_col,    # Model names on the x-axis
        y=r2_col,       # R^2 values on the y-axis
        data=df_sorted, 
        palette="viridis",  
        edgecolor=".2"
    )
    
    # Add R^2 values as labels on top of the bars
    for index, row in df_sorted.iterrows():
        # Text label is added on top of the bar (y=R2 value)
        ax.text(
            index,                                  # x position (index of the bar, 0, 1, 2...)
            row[r2_col] + 0.01,                     # y position (R2 value + slight offset above the bar)
            f'{row[r2_col]:.4f}',                   # text to display
            color='black', 
            ha="center",                            # center the text horizontally over the bar
            va="bottom",                            # align the text to the bottom (just above the bar)
            fontsize=10
        )

    ax.set_title("Model Performance Comparison ($R^2$)", fontsize=16, fontweight='bold', pad=15)
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
    # The input_file argument has been removed and is now hardcoded as PARQUET_FILE_PATH

    # --- Parity Plot Arguments ---
    actual_col: str = typer.Option(
        "Actual", 
        "--actual-col", 
        "-a", 
        help="Column name for the actual (true) values in the Parquet file."
    ),
    predicted_col: str = typer.Option(
        "Predicted", 
        "--predicted-col", 
        "-p", 
        help="Column name for the predicted values in the Parquet file."
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
    
    # --- Bar Graph Arguments (Optional, derived from the same file) ---
    metrics_output_file: Path = typer.Option(
        "metrics_bar_plot.png",
        "--metrics-output-file",
        "-mo",
        help="Path to save the output PNG file for the optional bar plot."
    ),
    model_col: str = typer.Option(
        "Model",
        "--model-col",
        help="Column name for the model type in the Parquet file (for bar plot)."
    ),
    r2_col: str = typer.Option(
        "R_squared",
        "--r2-col",
        help="Column name for the R-squared value in the Parquet file (for bar plot)."
    ),
):
    """
    Loads hardcoded Parquet data, generates the parity plot, and optionally 
    generates the metrics bar plot if the required columns are present.
    """
    input_file = PARQUET_FILE_PATH # Use the hardcoded path
    
    try:
        # Check for scikit-learn dependency
        try:
            import sklearn.metrics
        except ImportError:
            typer.echo("Error: scikit-learn is required. Please install with 'pip install scikit-learn'.", err=True)
            raise typer.Exit(code=1)

        # Hardcoded file existence check
        if not input_file.exists():
             typer.echo(f"Error: Hardcoded input file not found at '{input_file.resolve()}'", err=True)
             raise typer.Exit(code=1)
        
        # 1. Load the single Parquet file
        print(f"Loading combined data from: {input_file.resolve()}")
        try:
            df_combined = pd.read_parquet(input_file)
        except ImportError:
            typer.echo("Error: To read Parquet, you need to install 'pyarrow' or 'fastparquet'. Try 'pip install pyarrow'.", err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(f"Error loading Parquet file: {e}", err=True)
            raise typer.Exit(code=1)


        # 2. Generate Parity Plot (Mandatory check)
        print("\n--- Generating Parity Plot ---")
        if actual_col not in df_combined.columns:
            typer.echo(f"Error: Actual column '{actual_col}' not found in the Parquet file.", err=True)
            raise typer.Exit(code=1)
        if predicted_col not in df_combined.columns:
            typer.echo(f"Error: Predicted column '{predicted_col}' not found in the Parquet file.", err=True)
            raise typer.Exit(code=1)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        generate_plot(
            df=df_combined,
            actual_col=actual_col,
            predicted_col=predicted_col,
            output_path=output_file,
            title=plot_title
        )

        # 3. Generate Bar Graph (Conditional check)
        print("\n--- Checking for Metrics Bar Plot Data ---")
        is_model_col_present = model_col in df_combined.columns
        is_r2_col_present = r2_col in df_combined.columns
        
        if is_model_col_present and is_r2_col_present:
            print("Required metric columns found. Generating bar plot...")
            
            metrics_output_file.parent.mkdir(parents=True, exist_ok=True)
            generate_bar_plot(
                df=df_combined,
                model_col=model_col,
                r2_col=r2_col,
                output_path=metrics_output_file
            )
        else:
            typer.echo(f"Skipping bar plot: Column '{model_col}' ({is_model_col_present}) or '{r2_col}' ({is_r2_col_present}) not found in the Parquet file.", err=True)
            
    except typer.Exit:
        # Pass through expected exits
        pass
    except Exception as e:
        typer.echo(f"An unexpected error occurred: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

