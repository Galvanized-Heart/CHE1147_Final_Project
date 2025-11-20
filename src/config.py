from pathlib import Path
from dataclasses import dataclass
import copy

from dotenv import load_dotenv
from loguru import logger
import numpy as np
from skopt.space import Integer, Real, Categorical

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DOWNLOAD_URL = "https://github.com/ChemBioHTP/EnzyExtract/raw/main/EnzyExtractDB/EnzyExtractDB_176463.parquet"
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "EnzyExtractDB_176463.parquet"
INTERIM_DATA_DIR = DATA_DIR / "interim"
INTERIM_DATA_PATH = INTERIM_DATA_DIR / "cleaned_data.parquet"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_data.parquet"
SPLITS_DIR = PROCESSED_DATA_DIR / 'splits'
EXTERNAL_DATA_DIR = DATA_DIR / "external"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CV_RESULTS_DIR = REPORTS_DIR / "cross-validation"
HPO_RESULTS_DIR = REPORTS_DIR / "hpo"

TEMP_LO = -10.0  # Minimum temperature in °C
TEMP_HI = 100.0  # Maximum temperature in °C
TEMP_MAX_UNCERTAINTY = 10.0  # Maximum allowed uncertainty in for temperature °C
PH_LO = 0.0  # Minimum pH
PH_HI = 14.0  # Maximum pH
PH_MAX_UNCERTAINTY = 2.0  # Maximum allowed uncertainty for pH

# Processing Hyperparameters
linear = lambda x: x
log = np.log
log1p = np.log1p
COLUMN_TRANSFORMS = {
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
    "instability_index": ("linear", linear),
    "kcat_value": ("log", log),
    "km_value": ("log", log),
}

BASIC_FEATURE_COLS = [
    "mol_wt", "log_p", "tpsa", "num_h_donors", 
    "num_h_acceptors", "num_rot_bonds", "seq_length", 
    "seq_mol_wt", "pI", "aromaticity", "instability_index"
]
TEMP_PH_FEATURE_COLS = ["pH_value", "temperature_value"]
ADVANCED_FEATURE_COLS = [f'morgan_{i}' for i in range(2048)] + [f'esm_{i}' for i in range(320)]
BASIC_TARGET_COLS = ["kcat_value", "km_value"]


@dataclass
class ExperimentConfig:
    normalize: bool
    use_trans: bool
    use_temp_ph: bool
    use_advanced: bool


def make_col_names_trans(cols):
    return [f'{transform_name}_{col_name}' for col_name, (transform_name, _) in COLUMN_TRANSFORMS.items() if col_name in cols]


def get_feature_cols(exp_config: ExperimentConfig) -> list:
    feature_cols = copy.deepcopy(BASIC_FEATURE_COLS)
    if exp_config.use_temp_ph:
        feature_cols += TEMP_PH_FEATURE_COLS

    if exp_config.use_trans:
        feature_cols = make_col_names_trans(feature_cols)

    advanced_cols = []
    if exp_config.use_advanced:
        advanced_cols = copy.deepcopy(ADVANCED_FEATURE_COLS)
    
    return (feature_cols, advanced_cols)


def get_target_cols(exp_config) -> list:
    target_cols = copy.deepcopy(BASIC_TARGET_COLS)
    if exp_config.use_trans:
        target_cols = make_col_names_trans(target_cols)
    
    return target_cols


NORM_TRANS_EXPERIMENT_COLS = {
    "no_temp_ph_no_advanced": ExperimentConfig(normalize=True, use_trans=True, use_temp_ph=False, use_advanced=False),
    "yes_temp_ph_no_advanced": ExperimentConfig(normalize=True, use_trans=True, use_temp_ph=True, use_advanced=False),
    "yes_temp_ph_yes_advanced": ExperimentConfig(normalize=True, use_trans=True, use_temp_ph=True, use_advanced=True),
}
PARITY_PLOT_EXPERIMENT_COLS = {
    "no_temp_ph_no_advanced": ExperimentConfig(normalize=True, use_trans=True, use_temp_ph=False, use_advanced=False),
    "yes_temp_ph_no_advanced": ExperimentConfig(normalize=True, use_trans=True, use_temp_ph=True, use_advanced=False),
    "yes_temp_ph_yes_advanced": ExperimentConfig(normalize=True, use_trans=True, use_temp_ph=True, use_advanced=True),
}
SHAP_EXPERIMENT_COLS = {
    "no_temp_ph_no_advanced": ExperimentConfig(normalize=True, use_trans=True, use_temp_ph=True, use_advanced=True),
}

# HPO search spaces
XGB_PARAM_SEARCH_SPACE = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
}

# CORRECTED: The keys now directly match the __init__ parameters of MLPWrapper
NN_PARAM_SEARCH_SPACE = {
    'hidden_layer_1': Integer(8, 128),
    'hidden_layer_2': Integer(8, 128),
    'activation': Categorical(['relu', 'tanh']),
    'solver': Categorical(['sgd']),
    'learning_rate_init': Real(1e-3, 1e-1, prior='log-uniform'),
    'max_iter': Integer(500, 1500),
    'early_stopping': Categorical([True]),
    'n_iter_no_change': Integer(20, 50),
}

HPO_SEARCH_SPACES = {
    "linear": {},
    "xgb": XGB_PARAM_SEARCH_SPACE,
    "nn": NN_PARAM_SEARCH_SPACE,
}

# HPO and Modeling Hyperparameters
HPO_ROUNDS = 5

# SHAP
SHAP_TEST_PCTG = 0.2

# Directories
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATA_OUTPUT_DIR = REPORTS_DIR / "data"

# Evaluation hyperparameters
MODELS_DIR = PROJ_ROOT / "models"
HPO_RESULTS_DIR = MODELS_DIR/ "hpo"
NUM_CROSSVAL_FOLDS = 2

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
