import json

def validate_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("JSON is valid!")
        return data  # Return the parsed data for further processing if needed
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")










# Example usage
input_file = "/home/maverick/Documents/MTech_sem3/SDE/Final_project/landslide_analysis/src/reddit_data_landslide/new_landslide_events_20241114_183847.json"
output_file = "/home/maverick/Documents/MTech_sem3/SDE/Final_project/landslide_analysis/src/reddit_data_landslide/new_landslide_events_20241114_183847.json"
# Example usage
file_path = "path_to_your_json_file.json"
validate_json(input_file)
