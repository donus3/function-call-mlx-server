from jsonschema import validate, ValidationError
from functools import reduce
import operator

def get_nested_value(data_dict, path):
    """Access a nested value in a dictionary using a path list."""
    return reduce(operator.getitem, path, data_dict)

def set_nested_value(data_dict, path, value):
    """Set a nested value in a dictionary using a path list."""
    get_nested_value(data_dict, path[:-1])[path[-1]] = value

def attempt_conversion(value, target_type):
    """
    Tries to convert a value to the target JSON schema type.
    Returns the converted value or raises an exception if not possible.
    """
    try:
        if target_type == "string":
            return str(value)
        if target_type == "number":
            return float(value)
        if target_type == "integer":
            # Must be a whole number, so convert to float first then int
            return int(float(value))
        if target_type == "boolean":
            # Handle common string representations of booleans
            if isinstance(value, str) and value.lower() in ["false", "no", "0"]:
                return False
            return bool(value)
        if target_type == "array" and not isinstance(value, list):
            # A simple healing strategy: wrap the single item in a list
            return [value]
    except (ValueError, TypeError) as e:
        # Re-raise with a more informative message
        raise TypeError(f"Could not convert '{value}' to type '{target_type}'.") from e
    
    # Return original value if no conversion was applicable (e.g., for 'object' or 'null')
    return value


def heal_from_schema(data, schema):
    """
    Validates a dictionary against a JSON schema and attempts to "heal"
    type errors by casting values to the correct type.
    """
    if schema == None:
        return data
    healed_data = data.copy() # Work on a copy to avoid modifying the original
    max_attempts = 20 # Safety break to prevent infinite loops
    
    for i in range(max_attempts):
        try:
            # Try to validate the data
            validate(instance=healed_data, schema=schema)
            # If successful, break the loop and return the result
            return healed_data
        except ValidationError as e:
            # Check if the error is a type mismatch
            if e.validator == "type":
                path_list = list(e.path)
                expected_type = e.validator_value
                incorrect_value = e.instance
                
                print(f" Mismatch at '{'.'.join(map(str, path_list))}':")
                print(f"  - Found: {incorrect_value} (type: {type(incorrect_value).__name__})")
                print(f"  - Expected type: '{expected_type}'")
                
                try:
                    # Attempt to convert to the correct type
                    converted_value = attempt_conversion(incorrect_value, expected_type)
                    set_nested_value(healed_data, path_list, converted_value)
                    print(f"  - ✨ Healed: Set value to {converted_value} (type: {type(converted_value).__name__})\n")
                except TypeError as conversion_error:
                    print(f"  - ❌ Healing Failed: {conversion_error}")
                    # If conversion fails, we cannot heal, so we stop.
                    raise
            else:
                # If it's another validation error (e.g., 'required', 'minLength'), we can't heal it.
                print(f"❌ Unhealable validation error: {e.message}")
                raise
    
    raise RuntimeError("Could not heal the data within the maximum number of attempts.")
