"""
This script is used for utility functions
"""
import json
from typing import List, Dict


def convert_value(value):
    """
    Convert the string value from the environment variable to the appropriate type.

    Args:
        value: The string value from the environment variable.

    Returns:
        The converted value.
    """
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    try:
        int_value = int(value)
        return int_value
    except ValueError:
        pass
    try:
        float_value = float(value)
        return float_value
    except ValueError:
        pass
    try:
        json_value = json.loads(value)
        return json_value
    except (ValueError, json.JSONDecodeError):
        pass
    return value

def count_non_empty_fields(test: str, list_ocr: List[Dict], list_asr: List[Dict]) -> int:
    # Đếm số lượng field không rỗng
    count = 0
    if test.strip():  # kiểm tra test không rỗng và không chứa toàn khoảng trắng
        count += 1
    if list_ocr:  # kiểm tra list_ocr không phải là danh sách rỗng
        count += 1
    if list_asr:  # kiểm tra list_asr không phải là danh sách rỗng
        count += 1
    return count
