from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import numpy as np

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

TEMP_LO = -10.0  # Minimum temperature in °C
TEMP_HI = 100.0  # Maximum temperature in °C
TEMP_MAX_UNCERTAINTY = 10.0  # Maximum allowed uncertainty in for temperature °C
PH_LO = 0.0  # Minimum pH
PH_HI = 14.0  # Maximum pH
PH_MAX_UNCERTAINTY = 2.0  # Maximum allowed uncertainty for pH

TEST_PERCENTAGE = 0.8  # Percentage of data to use for training
HPO_TEST_PERCENTAGE = 0.25 # Percentage of training data to reserve for validation set during hyperparam tuning
SPLIT_RANDOM_STATE = 7  # Random state for train-test split. 7 is the best number.
NUM_CROSSVAL_FOLDS = 5  # Number of folds for cross-validation

METRICS_DICT = {'train_mse': [], 'test_mse': [], 'train_mae': [], 'test_mae': [], 'train_r2': [], 'test_r2': []}

# XGBoost hyperparameters
XGB_RANDOM_STATE = 14
XGB_N_ESTIMATORS = 100 # Number of trees in XGBoost
XGB_MAX_DEPTH = 6 # Maximum depth of trees
XGB_LEARNING_RATE = 0.1 # Learning rate

# Neural Network hyperparameters
NN_RANDOM_STATE = 77
NN_HIDDEN_LAYER_SIZES = (64, 32)
NN_ACTIVATION = 'relu'
NN_SOLVER = 'adam'
NN_LEARNING_RATE_INIT = 0.001
NN_MAX_ITER = 500  # Max number of epochs
NN_EARLY_STOPPING = True
NN_N_ITER_NO_CHANGE = 20

# Processing Hyperparameters
linear = lambda x: x
log = np.log
log1p = np.log1p
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

MODELS_DIR = PROJ_ROOT / "models"
LINEAR_MODEL_PATH = MODELS_DIR / "linear_model.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
NN_MODEL_PATH = MODELS_DIR / "nn_model.pkl"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
