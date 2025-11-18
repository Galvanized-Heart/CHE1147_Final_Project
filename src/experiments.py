from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from sklearn.model_selection import train_test_split

from config import PROCESSED_DATA_PATH, NORM_TRANS_EXPERIMENT_COLS, PARITY_PLOT_EXPERIMENT_COLS, SHAP_EXPERIMENT_COLS, SHAP_TEST_PCTG, get_target_cols
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


def norm_trans_experiment():
    logger.info("Starting Normalization and Transformation Experiments")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    exp_results_dict = experiment_on_cols(df, NORM_TRANS_EXPERIMENT_COLS)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 2, figsize=(16, 22))
    axes = axes.flatten()

    plot_data = []

    for exp_name, results_by_model in exp_results_dict.items():
        exp_config = NORM_TRANS_EXPERIMENT_COLS[exp_name]
        target_names = get_target_cols(exp_config)
        for model_name, results in results_by_model.items():
            # 'val_mse' is now a list of lists: [[fold1_t1, f1_t2], [fold2_t1, f2_t2], ...]
            val_mse_per_fold = results['metrics'].val_mse
            
            # Calculate the mean MSE across folds for each target
            avg_mse_per_target = np.mean(val_mse_per_fold, axis=0)
            
            for i, mse in enumerate(avg_mse_per_target):
                target_name = target_names[i]
                
                plot_data.append({
                    "experiment": exp_name,
                    "model": model_name,
                    "target": target_name,
                    "val_mse": mse
                })
    
    df_plot = pd.DataFrame(plot_data)

    # 2. Create the 4x2 subplot grid
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 2, figsize=(18, 24), constrained_layout=True)
    axes = axes.flatten()

    all_exp_names = list(NORM_TRANS_EXPERIMENT_COLS.keys())
    
    for i, exp_name in enumerate(all_exp_names):
        ax = axes[i]
        
        # Filter data for the current experiment
        exp_df = df_plot[df_plot["experiment"] == exp_name]

        if exp_df.empty:
            ax.text(0.5, 0.5, 'No results for this experiment', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(exp_name, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Create the grouped bar plot
        bars = sns.barplot(
            data=exp_df,
            x="model",
            y="Validation MSE",
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

        ax.set_title(exp_name, fontsize=14)
        ax.set_ylabel("Validation MSE")
        ax.set_xlabel(None)
        
        # Adjust y-limit to make space for annotations
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], current_ylim[1] * 1.15)
        ax.legend(title='Target')



def parity_plot_experiment():
    logger.info("Starting Parity Plot Experiment")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    exp_results_dict = experiment_on_cols(df, PARITY_PLOT_EXPERIMENT_COLS)

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # We will plot for the 'yes_norm_yes_advanced' experiment
    exp_name = "yes_norm_yes_advanced"
    if exp_name not in exp_results_dict:
        # Fallback to the first available experiment if the target is not found
        exp_name = list(exp_results_dict.keys())[0]

    model_results = exp_results_dict[exp_name]
    models_to_plot = ['linear', 'xgb', 'nn']

    for ax, model_name in zip(axes, models_to_plot):
        parity_data = model_results[model_name]['parity_data']
        y_true = parity_data.y_true
        y_pred = parity_data.y_pred
        
        ax.scatter(y_true, y_pred, alpha=0.5, label="Predictions")
        
        # Add a diagonal line for reference
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Ideal")
        ax.set_aspect('equal', 'box')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ax.set_title(f"Parity Plot for {model_name.upper()} Model", fontsize=14)
        ax.set_xlabel("True Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.legend()

    plt.suptitle(f"Parity Plots for Experiment: {exp_name}", fontsize=16, y=1.03)
    plt.tight_layout()
    plt.savefig("parity_plots.png", dpi=300)
    logger.info("Saved parity plots to parity_plots.png")


def shap_experiment():
    logger.info("Starting SHAP Experiment")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    shap_results_dict = experiment_on_cols(df, SHAP_EXPERIMENT_COLS)

    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, 3, figsize=(25, 12))
    models_to_plot = ['linear', 'xgb', 'nn']
    
    target_names = SHAP_EXPERIMENT_COLS["yes_norm_yes_advanced"][3]

    for col_idx, model_name in enumerate(models_to_plot):
        shap_values_list = shap_results_dict[model_name]['shap_values']
        X_test_sample = shap_results_dict[model_name]['X_test_sample']

        for row_idx, target_name in enumerate(target_names):
            ax = axes[row_idx, col_idx]
            plt.sca(ax)  # Set the current axis for shap to use

            ax.set_title(f'SHAP Summary for {model_name.upper()}\nTarget: {target_name}')
            
            # Get the shap values for the current target
            current_shap_values = shap_values_list[row_idx]

            # Create the SHAP summary plot on the current axis
            shap.summary_plot(current_shap_values, X_test_sample, show=False)

    plt.suptitle("SHAP Feature Importance Summary", fontsize=20, y=1.0)
    # Use tight_layout and adjust rect to prevent the suptitle from overlapping plots
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("shap_summary_plots.png", dpi=300)
    logger.info("Saved SHAP summary plots to shap_summary_plots.png")



