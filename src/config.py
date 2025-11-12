from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DOWNLOAD_URL = "https://github.com/ChemBioHTP/EnzyExtract/raw/main/EnzyExtractDB/EnzyExtractDB_176463.parquet"
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "EnzyExtractDB_176463.parquet"
INTERIM_DATA_DIR = DATA_DIR / "interim"
INTERIM_DATA_PATH = INTERIM_DATA_DIR / "cleaned_data.parquet"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

TEMP_LO = -10.0  # Minimum temperature in °C
TEMP_HI = 100.0  # Maximum temperature in °C
TEMP_MAX_UNCERTAINTY = 10.0  # Maximum allowed uncertainty in for temperature °C
PH_LO = 0.0  # Minimum pH
PH_HI = 14.0  # Maximum pH
PH_MAX_UNCERTAINTY = 2.0  # Maximum allowed uncertainty for pH

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
