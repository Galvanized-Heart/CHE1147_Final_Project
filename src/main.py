from loguru import logger
import joblib

from config import PROCESSED_DATA_PATH, HPO_RESULTS_DIR, ADVANCED_FEATS_COLS, BASIC_FEATS_COLS, \
PH_FEATS_COLS, TEMPERATURE_FEATS_COLS, KCAT_TARGET_COLS, KM_TARGET_COLS
import pandas as pd
from dataset import download_and_clean_data
from features import process_data
from split import create_splits
from hpo import run_hpo
from modeling.train import run_kfold_validation

"""
FEATURE_TRANSFORMS = {
    "kcat_value": ("log", log),
    "km_value": ("log", log),
    "pH_value": ("linear", linear),
    "temperature_value": ("linear", linear),
    "mol_wt": ("log", log),
    "log_p": ("linear", linear),
    "tpsa": ("log1p", log1p),
    "num_h_donors": ("log1p", log1p),
    "num_h_acceptors": ("log1p", log1p),
    "num_rot_bonds": ("log1p", log1p),
    "seq_length": ("log", log),
    "seq_mol_wt": ("log", log),
    "pI": ("linear", linear),
    "aromaticity": ("linear", linear),
    "instability_index": ("linear", linear)
}
"""


def main() -> None:
    logger.info("Starting the Full ML Pipeline")

    # Data Preparation
    download_and_clean_data()
    process_data()

    splits_path_dict = create_splits(PROCESSED_DATA_PATH)
    logger.success("Data splitting complete.")

    # Define Experiments
    all_targets = KCAT_TARGET_COLS + KM_TARGET_COLS
    experiments = {
        "basic_features": {
            "features": BASIC_FEATS_COLS, 
            "targets": all_targets,
        },
        "basic_plus_conditions": {
            "features": BASIC_FEATS_COLS + PH_FEATS_COLS + TEMPERATURE_FEATS_COLS,
            "targets": all_targets,
        },
        "advanced_features": {
            "features": ADVANCED_FEATS_COLS, 
            "targets": all_targets,
        },
        "advanced_plus_conditions": {
            "features": ADVANCED_FEATS_COLS + PH_FEATS_COLS + TEMPERATURE_FEATS_COLS,
            "targets": all_targets,
        },
        "all_features": {
            "features": BASIC_FEATS_COLS + ADVANCED_FEATS_COLS + PH_FEATS_COLS + TEMPERATURE_FEATS_COLS,
            "targets": all_targets,
        }
    }

    all_summaries = {}

    # Loop Through Experiments for HPO and Training
    for experiment in experiments:
        exp_name = experiment["name"]
        feature_cols = experiment["features"]
        target_cols = experiment["targets"]
        
        logger.info(f"Running Pipeline for Experiment: {exp_name}")

        # Hyperparameter Optimization
        hpo_results_path = HPO_RESULTS_DIR / f'best_params_{exp_name}.joblib'
        
        if hpo_results_path.exists():
            logger.info(f"Found existing HPO results for {exp_name}. Loading from file.")
            best_hyperparameters = joblib.load(hpo_results_path)
        else:
            logger.warning(f"HPO results for {exp_name} not found. Starting HPO.")
            best_hyperparameters = run_hpo(
                splits_dict=splits_path_dict,
                feature_cols=feature_cols,
                experiment_name=exp_name
            )
            logger.success(f"HPO complete for {exp_name}.")

        # K-Fold Cross-Validation
        kfold_summary = run_kfold_validation(
            splits_dict=splits_path_dict,
            best_params=best_hyperparameters,
            feature_cols=feature_cols,
            target_cols=target_cols,
            experiment_name=exp_name
        )
        all_summaries[exp_name] = kfold_summary

    # Report Results
    logger.success("All Experiments Completed")
    print("\n\n" + "="*50)
    print("      FINAL PERFORMANCE COMPARISON (val_r2 mean)")
    print("="*50)

    final_comparison = {}
    for exp_name, summary_df in all_summaries.items():
        final_comparison[exp_name] = summary_df[('val_r2', 'mean')]

    comparison_df = pd.DataFrame(final_comparison).round(4)
    print(comparison_df)
    print("="*50)
    logger.info("Pipeline finished.")

    # TODO: Predict full validation results for plots 

if __name__ == "__main__":
    main()