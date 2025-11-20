import os
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split

# Keep your existing config imports
from config import (
    PROCESSED_DATA_PATH, 
    NORM_TRANS_EXPERIMENT_COLS, 
    PARITY_PLOT_EXPERIMENT_COLS, 
    SHAP_EXPERIMENT_COLS, 
    SHAP_TEST_PCTG, 
    get_feature_cols, 
    get_target_cols, 
    DATA_OUTPUT_DIR
)
# Keep your existing modeling imports
from hpo import run_full_bayes_hpo
from modeling.train import single_experiment


def shap_analysis(X, y, model):
    """
    Helper function to compute SHAP values. 
    Returns (shap_values_array, X_test_sample_df).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SHAP_TEST_PCTG, random_state=0
    )
    model.fit(X_train, y_train)

    feature_names = X_train.columns.tolist()
    
    def predict_wrapper(X_array):
        X_df = pd.DataFrame(X_array, columns=feature_names)
        return model.predict(X_df)

    background_data = shap.sample(X_train, 100)
    explainer = shap.KernelExplainer(predict_wrapper, background_data)
    
    # Sample the test set for the actual explanation
    X_test_sample = shap.sample(X_test, min(100, len(X_test)))
    
    # shap_values for multi-output is usually a list of arrays or a 3D array
    shap_values = explainer.shap_values(X_test_sample)
    
    # Ensure consistency: Convert list of arrays to 3D array if necessary: (samples, features, targets)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)
        
    return shap_values, X_test_sample


def experiment_on_cols(df, cols_dict):
    """
    Helper function to run HPO and Training.
    """
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


# =============================================================================
# 1. Metrics Experiment (Temp/pH/Advanced) - Save Data
# =============================================================================
def generate_metrics_experiment_data():
    """
    Runs experiments and saves validation metrics (MSE, MAE, R2) to a CSV.
    
    Output columns: 
    [experiment, model, target, metric_type, metric_display_name, metric_value]
    """
    logger.info("Starting Metrics Data Generation...")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    # df = df.head(1000) # TODO: Uncomment for quick testing

    # Run the training pipeline
    exp_results_dict = experiment_on_cols(df, NORM_TRANS_EXPERIMENT_COLS)

    data_rows = []
    metrics_map = {
        'val_mse': 'Validation MSE',
        'val_mae': 'Validation MAE',
        'val_r2': 'Validation R²',
    }

    # Extract data from results structure
    for exp_name, results_by_model in exp_results_dict.items():
        exp_config = NORM_TRANS_EXPERIMENT_COLS[exp_name]
        target_names = get_target_cols(exp_config)
        
        for model_name, results in results_by_model.items():
            for attr_name, metric_display_name in metrics_map.items():
                # metric_per_fold shape: (n_folds, n_targets)
                metric_per_fold = getattr(results['metrics'], attr_name)
                
                # Average across folds to get a single scalar per target
                avg_metric_per_target = np.mean(metric_per_fold, axis=0)
                
                for i, metric_value in enumerate(avg_metric_per_target):
                    data_rows.append({
                        "experiment": exp_name,
                        "model": model_name,
                        "target": target_names[i],
                        "metric_type": attr_name,
                        "metric_display_name": metric_display_name,
                        "metric_value": metric_value,
                    })
    
    # Save to disk
    df_out = pd.DataFrame(data_rows)
    save_path = DATA_OUTPUT_DIR / "metrics_experiment_data.csv"
    df_out.to_csv(save_path, index=False)
    logger.success(f"Saved metrics data to {save_path}")


# =============================================================================
# 2. Parity Plot Experiment - Save Data
# =============================================================================
def generate_parity_experiment_data():
    """
    Runs experiments and saves True vs Predicted values to a CSV.
    
    Output columns: 
    [experiment, model, target, sample_index, y_true, y_pred]
    """
    logger.info("Starting Parity Data Generation...")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    # df = df.head(1000) # TODO: Uncomment for quick testing

    # Run the training pipeline
    exp_results_dict = experiment_on_cols(df, PARITY_PLOT_EXPERIMENT_COLS)

    if not exp_results_dict:
        logger.warning("No parity plot experiments were run.")
        return

    data_rows = []
    
    for exp_name, model_results in exp_results_dict.items():
        exp_config = PARITY_PLOT_EXPERIMENT_COLS[exp_name]
        target_names = get_target_cols(exp_config)

        for model_name, res in model_results.items():
            # parity_data object contains y_true and y_pred arrays
            parity_obj = res['parity_data']
            
            # Convert to dataframes to easily handle multi-output columns
            y_true_df = pd.DataFrame(parity_obj.y_true, columns=target_names)
            y_pred_df = pd.DataFrame(parity_obj.y_pred, columns=target_names)

            # Iterate over each target to flatten the data structure
            for t_col in target_names:
                t_true = y_true_df[t_col].values
                t_pred = y_pred_df[t_col].values
                
                # Add every sample point to the list
                for idx in range(len(t_true)):
                    data_rows.append({
                        "experiment": exp_name,
                        "model": model_name,
                        "target": t_col,
                        "sample_index": idx,
                        "y_true": t_true[idx],
                        "y_pred": t_pred[idx]
                    })

    # Save to disk
    df_out = pd.DataFrame(data_rows)
    save_path = DATA_OUTPUT_DIR / "parity_experiment_data.csv"
    df_out.to_csv(save_path, index=False)
    logger.success(f"Saved parity data to {save_path}")


# =============================================================================
# 3. SHAP Experiment - Save Data
# =============================================================================
def generate_shap_experiment_data():
    """
    Runs SHAP analysis and saves a long-format DataFrame suitable for plotting.
    
    Output columns: 
    [model, target, sample_id, feature, feature_value, shap_value]
    """
    logger.info("Starting SHAP Data Generation...")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    # df = df.head(1000) # TODO: Uncomment for quick testing

    if len(SHAP_EXPERIMENT_COLS) != 1:
        raise ValueError("SHAP_EXPERIMENT_COLS should contain exactly one experiment configuration.")
    
    exp_name = list(SHAP_EXPERIMENT_COLS.keys())[0]
    exp_config = SHAP_EXPERIMENT_COLS[exp_name]
    
    # Run Training
    exp_run_results = experiment_on_cols(df, SHAP_EXPERIMENT_COLS)[exp_name]

    models_to_run = ['linear', 'xgb', 'nn']
    
    feature_cols, _ = get_feature_cols(exp_config)
    target_cols = get_target_cols(exp_config)
    X = df[feature_cols]
    y = df[target_cols]
    
    all_shap_rows = []

    for model_name in models_to_run:
        if model_name not in exp_run_results: 
            continue

        logger.info(f"Calculating SHAP for {model_name}...")
        final_model = exp_run_results[model_name]['model']
        
        # shap_values_3d shape: (n_samples, n_features, n_targets)
        # X_test_sample shape: (n_samples, n_features)
        shap_values_3d, X_test_sample = shap_analysis(X, y, final_model)
        
        # Reset index to ensure we can merge on ID later
        X_test_sample = X_test_sample.reset_index(drop=True)
        
        # Iterate through targets to flatten the 3D array
        for t_idx, target_name in enumerate(target_cols):
            
            # 1. Extract SHAP matrix for this specific target: (n_samples, n_features)
            shap_matrix = shap_values_3d[:, :, t_idx]
            
            # 2. Create a DataFrame for SHAP values
            df_shap_vals = pd.DataFrame(shap_matrix, columns=X_test_sample.columns)
            df_shap_vals['sample_id'] = df_shap_vals.index
            
            # 3. Create a DataFrame for Feature values (the actual X values)
            df_feat_vals = X_test_sample.copy()
            df_feat_vals['sample_id'] = df_feat_vals.index
            
            # 4. Melt SHAP values -> [sample_id, feature, shap_value]
            melted_shap = df_shap_vals.melt(
                id_vars='sample_id', 
                var_name='feature', 
                value_name='shap_value'
            )
            
            # 5. Melt Feature values -> [sample_id, feature, feature_value]
            melted_feat = df_feat_vals.melt(
                id_vars='sample_id', 
                var_name='feature', 
                value_name='feature_value'
            )
            
            # 6. Merge them together
            merged_df = pd.merge(melted_shap, melted_feat, on=['sample_id', 'feature'])
            
            # 7. Add Metadata
            merged_df['model'] = model_name
            merged_df['target'] = target_name
            
            all_shap_rows.append(merged_df)

    # Combine all models/targets into one big DataFrame
    final_df = pd.concat(all_shap_rows, ignore_index=True)
    
    # Save to CSV (Using CSV for compatibility, but parquet is better for size if available)
    save_path = DATA_OUTPUT_DIR / "shap_experiment_data.csv"
    final_df.to_csv(save_path, index=False)
    logger.success(f"Saved SHAP data to {save_path}")


