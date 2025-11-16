from typing import Tuple
import re

import pandas as pd


def parse_pH(input_string: str, verbose: bool = False) -> Tuple[float, float]:
	"""
	Parses a string to find a pH value and its uncertainty.
	Returns (value, uncertainty) or (None, None).
	"""
	if pd.isna(input_string):
		# If str is None, return (None, None)
		return (None, None)

	input_string = str(input_string) # Ensure it's a string

	if re.search(r'\b(physiological conditions)\b', input_string, re.IGNORECASE):
		# Standard physiological range is 7.35-7.45
		return (7.4, 0.05)

	if re.search(r'\b(H2O|neutral)\b', input_string, re.IGNORECASE):
		# If str is a keyword for pH around 7, return 7.0 ± 0.5
		return (7.0, 0.5)

	if re.search(r'various', input_string, re.IGNORECASE):
		# If str contains various in any capacity, return (None, None)
		if verbose:
			print(f"Keyword 'various' found in: '{input_string}'. Skipping.")
		return (None, None)

	# Case: '7.10 ± 0.03'
	match = re.search(r'(\d+\.?\d*)\s*(?:±|\+/-)\s*(\d+\.?\d*)', input_string)
	if match:
		value = float(match.group(1))
		uncertainty = float(match.group(2))
		# Check if the value is in a valid pH range
		if 0 <= value <= 14:
			return (value, uncertainty)

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
		# If 1 valid number, return number with 0 uncertainty
		return (valid_numbers[0], 0.0)
	elif num_count == 2:
		# If 2 valid numbers (a range, e.g., "pH 7-8"), return average and half-range
		difference = abs(valid_numbers[0] - valid_numbers[1])
		value = sum(valid_numbers) / 2
		uncertainty = difference / 2
		return (value, uncertainty)
	else:
		# This block handles the cases where num_count is 0 or > 2.
		if verbose:
			if num_count == 0:
					print(f"No valid numbers (0-14) found in: '{input_string}'")
			else: # num_count > 2
					print(f"Too many valid numbers ({num_count}) found in: '{input_string}'")
		return (None, None)


def parse_temperature(temp_str: str, verbose: bool = False) -> Tuple[float, float]:
    """
    Returns (Value, Uncertainty)
    """

    if pd.isna(temp_str):
        return (None, None)

    temp_str = str(temp_str).strip()
    temp_str_lower = temp_str.lower()

    # 16. '5 / pH 8.6' -> (5.0, 0.0)
    # 23. '54/pH 7' -> (54.0, 0.0)
    match = re.match(r'(\d+\.?\d*)\s*/\s*ph\s*[\d\.]+', temp_str_lower)
    if match:
        return (float(match.group(1)), 0.0)

    # 15. 'pH 5.5, 30' -> (30.0, 0.0)
    # 21. 'pH 7.3 40' -> (40.0, 0.0)
    match = re.search(r'ph\s*[\d\.]+\s*[:,]?\s*(\d+\.?\d*)', temp_str_lower)
    if match:
        return (float(match.group(1)), 0.0)

    # 17. 'AT 18°C' -> (18.0, 0.0)
    # 18. '3 h at -37' -> (-37.0, 0.0)
    # 19. '20 min at 37' -> (37.0, 0.0)
    # 20. 'incubated at 37 for 60 min' -> (37.0, 0.0)
    match = re.search(r'(?:at|for)\s*(\-?\d+\.?\d*)', temp_str_lower)
    if match:
        return (float(match.group(1)), 0.0)

    # 22. 'Approximately 23°C' -> (23.0, 0.0)
    # 30. approx. 20°C -> (20.0, 0.0)
    match = re.search(r'(?:approximately|approx|~)\s*(\d+\.?\d*)', temp_str_lower)
    if match:
        # Note: While "approximately" implies uncertainty, it's not quantified.
        # Returning 0.0 for consistency with other single-value parses.
        return (float(match.group(1)), 0.0)

    # 27. 'optimal temperature 50°C' -> (50.0, 0.0)
    # 28. 'optimal at '37°C' -> (37.0, 0.0)
    # 29. 'Optimal activity at '37°C' -> (37.0, 0.0)
    match = re.search(r'(?:optimal temperature|optimal at|optimal activity at)\s*\'?(\d+\.?\d*)', temp_str_lower)
    if match:
      return (float(match.group(1)), 0.0)

    # Handle room temperature
    if "room temperature" in temp_str_lower or "rt" in temp_str_lower or "ambient" in temp_str_lower or "average temperature" in temp_str_lower or 'r.t' in temp_str_lower:
        # Assuming room temperature is 20-25°C -> 22.5 ± 2.5
        return (22.5, 2.5)

    # 1. '18°C' -> (18.0, 0.0) | '-45°C' -> (-45.0, 0.0)
    # 4. '22 ± 2 °C' -> (22.0, 2.0)
    # 6. '22°C ± 1°C' -> (22.0, 1.0)
    # 12. '25, pH 4.5' -> (25.0, 0.0)
    # 13. '25.0 ± 0.1' -> (25.0, 0.1)
    # Updated regex to capture uncertainty from '±'
    match = re.match(r'(\-?\d+\.?\d*)\s*(?:°c)?\s*(?:±\s*(\d+\.?\d*))?\s*(?:°c|,\s*ph.*|$)', temp_str_lower)
    if match:
        value = float(match.group(1))
        uncertainty = float(match.group(2)) if match.group(2) else 0.0
        return (value, uncertainty)

    # 2. '18°C to 33°C' -> (25.5, 7.5)
    # 5. '22-23°C' -> (22.5, 0.5)
    # 7. '23 to 25°C' -> (24.0, 1.0)
    # 11. '24 to 28' -> (26.0, 2.0)
    # 14. '55–60°C' -> (57.5, 2.5)
    # 24. '-10 to 2 °C' -> (-4.0, 6.0)
    # 26. '30/50°C' -> (40.0, 10.0)
    # Updated to return (midpoint, half-range)
    match = re.match(r'(\-?\d+\.?\d*)\s*(?:°c)?\s*(?:to|-|–|/)\s*(\-?\d+\.?\d*)\s*(?:°c)?', temp_str_lower)
    if match:
        temp1 = float(match.group(1))
        temp2 = float(match.group(2))
        value = (temp1 + temp2) / 2
        uncertainty = abs(temp2 - temp1) / 2
        return (value, uncertainty)

    # 3. '183K' -> (-90.15, 0.0)
    # 8. '245 K' -> (-28.15, 0.0)
    # 9. '248 K to 343 K' -> (22.35, 47.5)
    # Updated to return (midpoint, half-range) for K ranges
    match = re.match(r'(\d+\.?\d*)\s*k(?:\s*(?:to|-|–)\s*(\d+\.?\d*)\s*k)?', temp_str_lower)
    if match:
        temp_k1 = float(match.group(1))
        if match.group(2):
            temp_k2 = float(match.group(2))
            value_k = (temp_k1 + temp_k2) / 2
            uncertainty_k = abs(temp_k2 - temp_k1) / 2
            # Uncertainty is the same in K as in °C
            return (value_k - 273.15, uncertainty_k)
        else:
            value_k = temp_k1
            return (value_k - 273.15, 0.0)

    # 10. '22' -> (22.0, 0.0)
    match = re.match(r'^\-?\d+\.?\d*$', temp_str_lower)
    if match:
        return (float(temp_str_lower), 0.0)

    # TODO:
    # heated to 60
    # optimal T 70
    # 28°-30°C
    # optimal T 30°C
    # optimal 40.0 ± 3°C
    # 25 ºC

    if verbose:
        print(f"Failed to parse temperature: {temp_str}")

    return (None, None)


