from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from sklearn.model_selection import train_test_split

from config import PROCESSED_DATA_PATH, NORM_TRANS_EXPERIMENT_COLS, PARITY_PLOT_EXPERIMENT_COLS, SHAP_EXPERIMENT_COLS, SHAP_TEST_PCTG, get_target_cols, FIGURES_DIR
from hpo import run_full_bayes_hpo
from modeling.train import single_experiment


def shap_analysis(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SHAP_TEST_PCTG, random_state=42
    )
    model.fit(X_train, y_train)
    explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
    X_test_sample = shap.sample(X_test, min(100, len(X_test)))
    shap_values = explainer.shap_values(X_test_sample)

    return (shap_values, X_test_sample)


def experiment_on_cols(df, cols_dict):
    exp_results_dict = {}
    for (exp_name, exp_config) in cols_dict.items():
        logger.info(f"Experiment: {exp_name}")

        logger.info("Starting HPO")
        best_params_dict = run_full_bayes_hpo(
            config_name=exp_name,
            df=df,
            exp_config=exp_config,
        )

        logger.info("Starting Single Experiment with Best Params")
        exp_results = single_experiment(
            df=df,
            exp_config=exp_config,
            xgb_params=best_params_dict['xgb'],
            nn_params=best_params_dict['nn'],
        )

        exp_results_dict[exp_name] = exp_results
    
    return exp_results_dict


def temp_ph_advanced_experiment():
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    df = df.iloc[:, 1000] # TODO: Remove!
    exp_results_dict = experiment_on_cols(df, NORM_TRANS_EXPERIMENT_COLS)

    plot_data = []
    metrics_to_plot = {
        'val_mse': 'Validation MSE',
        'val_mae': 'Validation MAE',
        'val_r2': 'Validation R²',
    }

    exp_names_dict = {
        "no_temp_ph_no_advanced": "No Temp/pH or Advanced Features",
        "yes_temp_ph_no_advanced": "Temp/pH, No Advanced Features",
        "yes_temp_ph_yes_advanced": "Temp/pH and Advanced Features"
    }

    for exp_name, results_by_model in exp_results_dict.items():
        exp_config = NORM_TRANS_EXPERIMENT_COLS[exp_name]
        target_names = get_target_cols(exp_config)
        for model_name, results in results_by_model.items():
            for attr_name, plot_title in metrics_to_plot.items():
                metric_per_fold = getattr(results['metrics'], attr_name)
                
                avg_metric_per_target = np.mean(metric_per_fold, axis=0)
                
                for i, metric_value in enumerate(avg_metric_per_target):
                    plot_data.append({
                        "experiment": exp_name,
                        "model": model_name,
                        "target": target_names[i],
                        "metric_name": plot_title,
                        "metric_value": metric_value,
                    })
    
    df_plot = pd.DataFrame(plot_data)

    # --- 3. Create the N x M Subplot Grid ---
    all_exp_names = list(NORM_TRANS_EXPERIMENT_COLS.keys())
    all_metric_names = list(metrics_to_plot.values())
    
    N = len(all_exp_names)
    M = len(all_metric_names)

    sns.set_theme(style="whitegrid")
    # Dynamically set figsize based on grid size
    fig, axes = plt.subplots(N, M, figsize=(M * 5.5, N * 5), constrained_layout=True)
    
    # Ensure axes is always a 2D array for consistent indexing, even if N=1 or M=1
    axes = np.atleast_2d(axes)

    for row_idx, exp_name in enumerate(all_exp_names):
        for col_idx, metric_name in enumerate(all_metric_names):
            ax = axes[row_idx, col_idx]
            
            # Filter the DataFrame for the specific data for this subplot
            exp_df = df_plot[
                (df_plot["experiment"] == exp_name) & 
                (df_plot["metric_name"] == metric_name)
            ]

            if exp_df.empty:
                ax.text(0.5, 0.5, 'No results', ha='center', va='center', transform=ax.transAxes)
            else:
                # Create the grouped bar plot for this cell
                bars = sns.barplot(
                    data=exp_df,
                    x="model",
                    y="metric_value",
                    hue="target",
                    ax=ax,
                    palette="muted"
                )
                
                # Add value annotations on top of the bars
                for bar in bars.patches:
                    ax.annotate(format(bar.get_height(), '.3f'),
                                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                ha='center', va='center',
                                size=9, xytext=(0, 8),
                                textcoords='offset points')
                
                # Adjust y-limit to make space for annotations
                current_ylim = ax.get_ylim()
                ax.set_ylim(current_ylim[0], current_ylim[1] * 1.15)
                ax.legend(title='Target')

            # --- Set Titles and Labels ---
            # Column titles are the metric names (only for the top row)
            if row_idx == 0:
                ax.set_title(metric_name, fontsize=14, weight='bold')

            # Row titles are the experiment names (only for the first column)
            if col_idx == 0:
                ax.set_ylabel(exp_names_dict[exp_name], fontsize=14, weight='bold')
            else:
                ax.set_ylabel(None)
            
            ax.set_xlabel(None)

    fig.suptitle("Model Performance Across Different Input Features", fontsize=20)
    fig_save_path = FIGURES_DIR / "temp_ph_advanced_experiment_grid.png"
    plt.savefig(fig_save_path, dpi=300)
    logger.info(f"Saved experiment grid plot to {fig_save_path}")
    plt.show()


def parity_plot_experiment():
    """
    Runs parity plot experiments and generates a 1xM grid of subplots,
    where M is the number of experiments. Each subplot shows the parity
    data for all models for that specific experiment.
    """
    logger.info("Starting Parity Plot Experiment")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    df = df.head(1000)
    
    # --- 1. Run the Experiments ---
    exp_results_dict = experiment_on_cols(df, PARITY_PLOT_EXPERIMENT_COLS)

    if not exp_results_dict:
        logger.warning("No parity plot experiments were run. Skipping plot generation.")
        return

    # --- 2. Create the 1 x M Subplot Grid ---
    all_exp_names = list(exp_results_dict.keys())
    M = len(all_exp_names)

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, M, figsize=(M * 6.5, 6), constrained_layout=True)

    # Ensure 'axes' is always an array, even if M=1, for consistent looping
    axes = np.atleast_1d(axes)

    models_to_plot = ['linear', 'xgb', 'nn']
    # Define colors to distinguish the models in each subplot
    model_colors = {'linear': 'C0', 'xgb': 'C2', 'nn': 'C1'}

    # --- 3. Loop Through Experiments and Create a Subplot for Each ---
    for ax, exp_name in zip(axes, all_exp_names):
        model_results = exp_results_dict[exp_name]

        # Plot all three models on the same subplot (ax)
        for model_name in models_to_plot:
            if model_name not in model_results:
                continue

            parity_data = model_results[model_name]['parity_data']
            y_true = parity_data.y_true
            y_pred = parity_data.y_pred
            
            # Scatter plot for the current model with a specific color and label
            ax.scatter(
                y_true, y_pred, 
                alpha=0.5, 
                color=model_colors.get(model_name), 
                label=model_name.upper()
            )
        
        # --- 4. Set Diagonal Line and Limits After Plotting All Models ---
        # Get the final axis limits after all scatters have been plotted
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        
        # Determine the overall min and max to make the plot square
        min_val = min(xlims[0], ylims[0])
        max_val = max(xlims[1], ylims[1])
        lims = [min_val, max_val]
        
        # Add the diagonal "ideal" line
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Ideal")
        
        # Apply the final limits and settings
        ax.set_aspect('equal', 'box')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        # --- 5. Add Labels, Title, and Legend ---
        ax.set_title(f"Experiment: {exp_name}", fontsize=14)
        ax.set_xlabel("True Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.legend()

    fig.suptitle("Model Parity Plots Across Different Experiments", fontsize=18)
    plt.savefig("parity_plots_by_experiment.png", dpi=300)
    logger.info("Saved parity plots to parity_plots_by_experiment.png")
    plt.show()


def shap_experiment():
    """
    Runs SHAP analysis and generates a grid of summary plots.
    The grid has N rows (for N targets) and M columns (for M models).
    """
    logger.info("Starting SHAP Experiment")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    df = df.head(1000) # TODO: Remove!
    print("B:", df.columns.tolist())

    # --- 1. Get the Trained Models ---
    # We run the experiment to get the final models trained on all data.
    # We assume SHAP_EXPERIMENT_COLS contains exactly one experiment configuration.
    if len(SHAP_EXPERIMENT_COLS) != 1:
        raise ValueError("SHAP_EXPERIMENT_COLS should contain exactly one experiment configuration.")
    
    exp_name = list(SHAP_EXPERIMENT_COLS.keys())[0]
    exp_config = SHAP_EXPERIMENT_COLS[exp_name]
    
    # This dictionary will contain the final fitted models under results['model']
    exp_run_results = experiment_on_cols(df, SHAP_EXPERIMENT_COLS)[exp_name]

    # --- 2. Perform SHAP Analysis for Each Model ---
    shap_data = {}
    models_to_plot = ['linear', 'xgb', 'nn']
    
    feature_cols, _ = get_feature_cols(exp_config) # Get basic and advanced features
    target_cols = get_target_cols(exp_config)
    X = df[feature_cols]
    y = df[target_cols]
    
    logger.info("Performing SHAP analysis on each model...")
    for model_name in models_to_plot:
        final_model = exp_run_results[model_name]['model']
        shap_values, X_test_sample = shap_analysis(X, y, final_model)
        shap_data[model_name] = {
            'shap_values': shap_values,
            'X_test_sample': X_test_sample
        }

    # --- 3. Create the N x M Subplot Grid ---
    N = len(target_cols)  # Number of targets = number of rows
    M = len(models_to_plot) # Number of models = number of columns

    sns.set_theme(style="white")
    fig, axes = plt.subplots(N, M, figsize=(M * 7, N * 5), constrained_layout=True)
    axes = np.atleast_2d(axes) # Ensure axes is always a 2D array

    # --- 4. Loop Through Grid and Create Plots ---
    for row_idx, target_name in enumerate(target_cols):
        for col_idx, model_name in enumerate(models_to_plot):
            ax = axes[row_idx, col_idx]
            
            # Retrieve the pre-calculated SHAP data for this model
            model_shap_data = shap_data[model_name]
            shap_values_list = model_shap_data['shap_values']
            X_test_sample = model_shap_data['X_test_sample']

            # CRITICAL: Select the SHAP values for the current target (row)
            # shap_values_list is a list where index corresponds to the target index
            current_target_shap_values = shap_values_list[row_idx]
            
            # Create the SHAP summary plot directly on the specified axis
            shap.summary_plot(
                current_target_shap_values, 
                X_test_sample, 
                ax=ax, 
                show=False
            )

            # --- Set Titles and Labels for Clarity ---
            # Set model names as column titles only for the top row
            if row_idx == 0:
                ax.set_title(model_name.upper(), fontsize=16, weight='bold')

            # Set target names as row titles only for the first column
            if col_idx == 0:
                # Use a text object for better placement than ylabel
                fig.text(0, (N - row_idx - 0.5) / N, target_name, 
                         ha='center', va='center', rotation='vertical', 
                         fontsize=16, weight='bold')

    fig.suptitle("SHAP Feature Importance Summary", fontsize=20)
    plt.savefig("shap_summary_plots_grid.png", dpi=300)
    logger.info("Saved SHAP summary plots to shap_summary_plots_grid.png")
    plt.show()



