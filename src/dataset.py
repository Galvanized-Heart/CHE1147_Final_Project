import os
import requests
import shutil
from pathlib import Path
import re

import pandas as pd
from loguru import logger

from config import DATA_DOWNLOAD_URL, RAW_DATA_DIR, RAW_DATA_PATH, INTERIM_DATA_DIR, INTERIM_DATA_PATH, \
    TEMP_HI, TEMP_LO,TEMP_MAX_UNCERTAINTY, PH_HI, PH_LO, PH_MAX_UNCERTAINTY
from parse import parse_pH, parse_temperature


def download_and_clean_data() -> None:
    if os.path.exists(INTERIM_DATA_PATH):
        logger.info(f"Interim data already exists at {INTERIM_DATA_PATH}. Skipping download and cleaning.")
        return

    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        logger.info(f"Downloading raw data from {DATA_DOWNLOAD_URL} to {RAW_DATA_PATH}")
        with requests.get(DATA_DOWNLOAD_URL, stream=True) as r:
            r.raise_for_status() 
            with open(RAW_DATA_PATH, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
                logger.info(f"Download complete. Data saved to {RAW_DATA_PATH}")
    
    logger.info(f"Loading raw data from {RAW_DATA_PATH}")
    df = pd.read_parquet(RAW_DATA_PATH)

    logger.info("Cleaning data")
    # Keep only relevant columns for the regression task
    relevant_cols = ["sequence", "smiles", "temperature", "pH", "kcat_value", "km_value"]
    df = df[relevant_cols].copy()

    # Parse pH and temperature columns
    ph_col_names = ["pH_value", "pH_uncertainty"]
    df[ph_col_names] = df['pH'].apply(parse_pH).apply(pd.Series)
    temp_col_names = ["temperature_value", "temperature_uncertainty"]
    df[temp_col_names] = df['temperature'].apply(parse_temperature).apply(pd.Series)

    # Drop temperature outliers
    df = df[df['temperature_value'] <= TEMP_HI]
    df = df[df['temperature_value'] >= TEMP_LO]

    # Drop pH outlier
    df = df[df['pH_value'] <= PH_HI]
    df = df[df['pH_value'] >= PH_LO]
    
    # Drop uncertainty outliers
    df = df[df['temperature_uncertainty'].fillna(0) <= TEMP_MAX_UNCERTAINTY]
    df = df[df['pH_uncertainty'].fillna(0) <= PH_MAX_UNCERTAINTY]

    # Drop uncertainty columns since we no longer need them
    df = df.drop(columns=["temperature_uncertainty", "pH_uncertainty"])

    # Drop rows where kcat is less than or equal to 0
    df = df[df['kcat_value'] > 0]

    # Drop rows where km is less than or equal to 0
    df = df[df['km_value'] > 0]

    # Reset index after filtering
    df = df.reset_index(drop=True)

    logger.info(f"Cleaned data has the following columns: {df.columns.tolist()}")
    logger.info(f"Cleaned data has {df.shape[0]} rows")

    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    logger.info(f"Saving cleaned interim data to {INTERIM_DATA_PATH}")
    df.to_parquet(INTERIM_DATA_PATH, index=False)

