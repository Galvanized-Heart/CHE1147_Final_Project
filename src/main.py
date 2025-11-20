from loguru import logger
import joblib

import pandas as pd
from dataset import download_and_clean_data
from features import process_data

#from experiments import temp_ph_advanced_experiment, parity_plot_experiment, shap_experiment
from src.experiments import generate_metrics_experiment_data, generate_parity_experiment_data, generate_shap_experiment_data


def main() -> None:
    logger.info("Starting the Full ML Pipeline")

    # Data Preparation
    download_and_clean_data()
    process_data()

    # Run Experiments
    generate_metrics_experiment_data()
    generate_parity_experiment_data()
    generate_shap_experiment_data()


if __name__ == "__main__":
    main()