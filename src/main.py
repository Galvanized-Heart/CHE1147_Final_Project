from config import PROCESSED_DATA_PATH
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
    #download_and_clean_data()
    #process_data()
    splits_path_dict = create_splits(PROCESSED_DATA_PATH)
    best_hyperparameters = run_hpo(splits_path_dict)
    kfold_summary = run_kfold_validation(splits_path_dict, best_hyperparameters)

if __name__ == "__main__":
    main()