from pathlib import Path
from loguru import logger
from typing import Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split, KFold

from config import PROCESSED_DATA_PATH, PROCESSED_DATA_DIR, SPLITS_DIR, TEST_PERCENTAGE, HPO_TEST_PERCENTAGE, \
    SPLIT_RANDOM_STATE, NUM_CROSSVAL_FOLDS



def create_splits(processed_data_path: Path, k: int = NUM_CROSSVAL_FOLDS, random_state: int = SPLIT_RANDOM_STATE) -> Dict[str, Any]:
    
    # Read data
    logger.info(f"Reading processed data from {processed_data_path}...")
    try:
        df = pd.read_parquet(processed_data_path)
        print(df.head())
    except FileNotFoundError:
        logger.error(f"Input file not found at {processed_data_path}. Please run the processing step first.")
        return {}
    X = df.drop(columns=['kcat_value', 'km_value', 'log_kcat_value', 'log_km_value', 'temperature', 'pH', 'pH_value', 
                         'temperature_value', 'mol_wt', 'log_p', 'tpsa', 'num_h_donors', 'num_h_acceptors', 
                         'num_rot_bonds', 'seq_length', 'seq_mol_wt', 'pI', 'aromaticity', 'instability_index'])
    y = df[['log_kcat_value', 'log_km_value']]

    # Create split dir
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Splits will be saved to {SPLITS_DIR}")

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    splits_path_dict = {'folds': {}, 'hpo': {}}

    logger.info(f"Creating and saving {k} folds for cross-validation...")
    for fold_idx, (train_index, val_index) in enumerate(kf.split(X), 1):
        # Create a directory for the current fold
        fold_dir = SPLITS_DIR / f'fold_{fold_idx}'
        fold_dir.mkdir(exist_ok=True)
        
        # Get the data for the current fold using the indices
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Define paths for this fold's files
        paths = {
            'X_train': fold_dir / 'X_train.parquet',
            'y_train': fold_dir / 'y_train.parquet',
            'X_val': fold_dir / 'X_val.parquet',
            'y_val': fold_dir / 'y_val.parquet'
        }

        # Save the dataframes
        X_train.to_parquet(paths['X_train'])
        y_train.to_parquet(paths['y_train'])
        X_val.to_parquet(paths['X_val'])
        y_val.to_parquet(paths['y_val'])
        
        # Store paths in the main dictionary
        splits_path_dict['folds'][f'fold_{fold_idx}'] = {
            'train': (paths['X_train'], paths['y_train']),
            'val': (paths['X_val'], paths['y_val'])
        }

        # Special case for HPO split
        if fold_idx == 1:
            logger.info(f"Creating HPO split from Fold 1's training set...")
            
            # Create HPO directory
            hpo_dir = SPLITS_DIR / 'hpo'
            hpo_dir.mkdir(exist_ok=True)

            # Split the training set of the first fold further
            X_hpo_train, X_hpo_val, y_hpo_train, y_hpo_val = train_test_split(
                X_train, y_train, test_size=HPO_TEST_PERCENTAGE, random_state=random_state
            )
            
            logger.info(f"  - HPO: Train set size: {len(X_hpo_train)}, Test set size: {len(X_hpo_val)}")

            # Define paths for HPO files
            hpo_paths = {
                'X_train': hpo_dir / 'X_train.parquet',
                'y_train': hpo_dir / 'y_train.parquet',
                'X_val': hpo_dir / 'X_val.parquet',
                'y_val': hpo_dir / 'y_val.parquet'
            }

            # Save the HPO dataframes
            X_hpo_train.to_parquet(hpo_paths['X_train'])
            y_hpo_train.to_parquet(hpo_paths['y_train'])
            X_hpo_val.to_parquet(hpo_paths['X_val'])
            y_hpo_val.to_parquet(hpo_paths['y_val'])

            # Store HPO paths in the main dictionary
            splits_path_dict['hpo'] = {
                'train': (hpo_paths['X_train'], hpo_paths['y_train']),
                'val': (hpo_paths['X_val'], hpo_paths['y_val'])
            }
            logger.success(f"HPO split saved to {hpo_dir}")
            
    logger.success(f"K-Fold splitting complete. All files saved under {SPLITS_DIR}")
    return splits_path_dict


if __name__ == "__main__":
    split_path_dict = create_splits(PROCESSED_DATA_PATH)
    print(split_path_dict)