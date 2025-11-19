from loguru import logger
import joblib

import pandas as pd
from dataset import download_and_clean_data
from features import process_data

from experiments import temp_ph_advanced_experiment, parity_plot_experiment, shap_experiment


def main() -> None:
    logger.info("Starting the Full ML Pipeline")

    # Data Preparation
    download_and_clean_data()
    process_data()

    # Run Experiments
    #temp_ph_advanced_experiment()
    parity_plot_experiment()
    shap_experiment()


if __name__ == "__main__":
    main()