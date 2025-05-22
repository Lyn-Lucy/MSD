import json

input_file_path = "/home/llx/models/datasets--liuhaotian--LLaVA-Instruct-150K/llava_v1_5_mix665k.json"
output_file_path = "/home/llx/models/datasets--liuhaotian--LLaVA-Instruct-150K/llava_v1_5_mix665k_fixed.json"

def convert_id_to_string(data):
    """
    Recursively converts 'id' fields to strings.
    
    Args:
        data (dict or list): The JSON data structure to process.
        
    Returns:
        dict or list: The processed JSON data structure with all 'id' fields as strings.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'id':
                data[key] = str(value)
            else:
                convert_id_to_string(value)
    elif isinstance(data, list):
        for item in data:
            convert_id_to_string(item)
    return data

# Read the JSON file
with open(input_file_path, 'r', encoding='utf-8') as infile:
    try:
        data = json.load(infile)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from {input_file_path}: {e}")
        raise

# Convert all 'id' fields to strings
converted_data = convert_id_to_string(data)

# Write the modified data back to a new JSON file
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(converted_data, outfile, ensure_ascii=False, indent=2)

print(f"All 'id' fields have been converted to strings and saved to {output_file_path}.")