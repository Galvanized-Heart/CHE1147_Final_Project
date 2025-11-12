import re
from pathlib import Path
import pandas as pd
import typer
from loguru import logger

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR



def parse_pH(input_string: str, pH_diff_threshold: float = 1.0) -> float | None:
  if pd.isna(input_string):
    # If str is None, return None
    return None

  if re.search(r'\b(H2O|physiological conditions|neutral)\b', input_string, re.IGNORECASE):
    # If str is a keyword for pH around 7, return 7.0
    return 7.0

  if re.search(r'various', input_string, re.IGNORECASE):
    # If str contains various in any capacity, return None
    print(f"Keyword 'various' found in: '{input_string}'. Skipping.")
    return None

  # Strip away numbers from chemical names
  chem_strip_pattern = r"\b\d+,\d+['-]"
  chem_stripped_string = re.sub(chem_strip_pattern, '', input_string)

  # Strip away buffer and temperature with unit markers
  unit_strip_pattern = r'\b\d+(?:\.\d+)?\s*(?:M|mM|µM|°C|%)(?=\s|\b|$)'
  stripped_string = re.sub(unit_strip_pattern, '', chem_stripped_string)

  # Extract remaining unitless numbers (assumed to be pH)
  extract_pattern = r'\b\d+(?:\.\d+)?\b'
  found_numbers_str = re.findall(extract_pattern, stripped_string)

  # Convert number strings to float
  all_found_numbers = [float(n) for n in found_numbers_str]

  # Drop numbers that aren't within realistic pH range 0-14
  valid_numbers = [n for n in all_found_numbers if 0 <= n <= 14]

  # Apply valid number count logic
  num_count = len(valid_numbers)
  if num_count == 1:
    # If 1 valid number, return number
    return valid_numbers[0]
  elif num_count == 2:
    # If 2 valid numbers, return average or None depending on accepted threshold
    difference = abs(valid_numbers[0] - valid_numbers[1])
    if difference <= pH_diff_threshold:
      # The range is narrow enough, so we trust the average
      return sum(valid_numbers) / 2
    else:
      # The range is too wide to be a specific value, so we reject it
      print(f"Range too wide ({difference:.1f}) in '{input_string}'. Skipping.")
      return None
  else:
    # This block handles the cases where num_count is 0 or > 2.
    if num_count == 0:
        print(f"No valid numbers (0-14) found in: '{input_string}'")
    else: # num_count > 2
        print(f"Too many valid numbers ({num_count}) found in: '{input_string}'")
  return None



def parse_temperature(temp_str: str) -> float | None:
    if pd.isna(temp_str):
        return None

    temp_str = str(temp_str).strip()
    temp_str_lower = temp_str.lower()

    # 16. '5 / pH 8.6' -> 5
    # 23. '54/pH 7' -> 54
    match = re.match(r'(\d+\.?\d*)\s*/\s*ph\s*[\d\.]+', temp_str_lower)
    if match:
        return float(match.group(1))

    # 15. 'pH 5.5, 30' -> 30
    # 21. 'pH 7.3 40' -> 40
    match = re.search(r'ph\s*[\d\.]+\s*[:,]?\s*(\d+\.?\d*)', temp_str_lower)
    if match:
        return float(match.group(1))

    # 17. 'AT 18°C' -> 18
    # 18. '3 h at -37' -> -37
    # 19. '20 min at 37' -> 37
    # 20. 'incubated at 37 for 60 min' -> 37
    match = re.search(r'(?:at|for)\s*(\-?\d+\.?\d*)', temp_str_lower)
    if match:
        return float(match.group(1))

    # 22. 'Approximately 23°C' -> 23
    # 30. approx. 20°C -> 20
    match = re.search(r'(?:approximately|approx|~)\s*(\d+\.?\d*)', temp_str_lower)
    if match:
        return float(match.group(1))

    # 27. 'optimal temperature 50°C' -> 50
    # 28. 'optimal at '37°C' -> 37
    # 29. 'Optimal activity at '37°C' -> 37
    match = re.search(r'(?:optimal temperature|optimal at|optimal activity at)\s*\'?(\d+\.?\d*)', temp_str_lower)
    if match:
      return float(match.group(1))

    if "room temperature" in temp_str_lower or "rt" in temp_str_lower or "ambient" in temp_str_lower or "average temperature" in temp_str_lower or 'r.t' in temp_str_lower:
        return 22.5 # Assuming room temperature is around 22.5°C

    # 1. '18°C' -> 18, '-45°C' -> -45
    # 4. 22 ± 2 °C -> 22
    # 6. '22°C ± 1°C' -> 22
    # 12. '25, pH 4.5' -> 25
    # 13. '25.0 ± 0.1' -> 25
    match = re.match(r'(\-?\d+\.?\d*)\s*(?:°c|\s*±.*|\s*,\s*ph.*|$)', temp_str_lower)
    if match:
        return float(match.group(1))


    # 2. '18°C to 33°C' -> (18 + 33) / 2 = 25.5
    # 5. '22-23°C' -> 22.5
    # 7. '23 to 25°C' -> 24
    # 11. '24 to 28' -> 26
    # 14. '55–60°C' -> 57.5 (handling different dash)
    # 24. '-10 to 2 °C' -> -4
    # 26. '30/50°C' -> 40
    match = re.match(r'(\-?\d+\.?\d*)\s*(?:°c)?\s*(?:to|-|–|/)\s*(\-?\d+\.?\d*)\s*(?:°c)?', temp_str_lower)
    if match:
        temp1 = float(match.group(1))
        temp2 = float(match.group(2))
        return (temp1 + temp2) / 2

    # 3. '183K' -> -90.15
    # 8. '245 K' -> 245
    # 9. '248 K to 343 K' -> 295.5
    match = re.match(r'(\d+\.?\d*)\s*k(?:\s*(?:to|-|–)\s*(\d+\.?\d*)\s*k)?', temp_str_lower)
    if match:
        temp_k1 = float(match.group(1))
        if match.group(2):
            temp_k2 = float(match.group(2))
            return ((temp_k1 + temp_k2) / 2) - 273.15
        else:
            return temp_k1 - 273.15

    # 10. '22' -> 22
    match = re.match(r'^\-?\d+\.?\d*$', temp_str_lower)
    if match:
        return float(temp_str_lower)
    return None



app = typer.Typer()

@app.command()
def main(
    input_path: Path = typer.Option(RAW_DATA_DIR / "EnzyExtractDB_176463.parquet", help="Path to the raw EnzyExtract dataset."),
    output_path: Path = typer.Option(INTERIM_DATA_DIR / "cleaned_data.parquet", help="Path to save the cleaned interim data."),
):
    """
    Loads the raw dataset, applies cleaning and parsing for temperature and pH,
    filters out invalid rows, and saves the result to the interim directory.
    """
    logger.info("Starting data cleaning process...")
    logger.info(f"Loading raw data from {input_path}")
    df = pd.read_parquet(input_path)

    logger.info("Selecting relevant columns and renaming...")
    df_filt = df[["sequence", "smiles", "temperature", "pH", "kcat_value", "km_value"]].copy()
    df_filt = df_filt.rename(columns={'kcat_value': 'kcat (s^{-1})', 'km_value': 'km (M)',
                                      'temperature': 'temperature_raw', 'pH': 'pH_raw'})

    logger.info("Parsing pH and temperature columns...")
    df_filt['pH'] = df_filt['pH_raw'].apply(parse_pH)
    df_filt['temperature'] = df_filt['temperature_raw'].apply(parse_temperature)

    logger.info("Filtering out rows with special amino acids...")
    rows_to_drop_mask = df_filt['sequence'].str.contains(r'\(.*\)', na=False)
    logger.info(f"Dropping {rows_to_drop_mask.sum()} rows with special amino acids.")
    df_filt = df_filt[~rows_to_drop_mask]

    logger.info("Dropping rows with missing essential data (sequence, smiles, targets, etc.)...")
    initial_rows = len(df_filt)
    df_filt.dropna(subset=['sequence', 'smiles', 'kcat (s^{-1})', 'km (M)', 'pH', 'temperature'], inplace=True)
    logger.info(f"Dropped {initial_rows - len(df_filt)} rows with missing values.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filt.to_parquet(output_path, index=False)
    logger.success(f"Cleaning complete. Cleaned data saved to {output_path}")

if __name__ == "__main__":
    app()