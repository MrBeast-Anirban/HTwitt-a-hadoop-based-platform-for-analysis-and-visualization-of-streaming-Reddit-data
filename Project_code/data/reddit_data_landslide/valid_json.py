import json

# Read the incomplete JSON file
with open('part-00000-a5d41d4f-874e-4826-bdcb-fb67d9c4f936-c000.json', 'r') as file:
    lines = file.readlines()

# Wrap the JSON objects in an array and separate them with commas
json_content = '[' + ','.join(line.strip() for line in lines) + ']'

# Parse the JSON to ensure it's valid
try:
    json_data = json.loads(json_content)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    exit(1)

# Write the valid JSON back to the file
with open('part-00000-a5d41d4f-874e-4826-bdcb-fb67d9c4f936-c000.json', 'w') as file:
    json.dump(json_data, file, indent=4)

print("JSON file has been made valid and saved.")