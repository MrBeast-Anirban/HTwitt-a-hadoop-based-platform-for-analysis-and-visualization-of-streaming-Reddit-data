from collections import defaultdict
import json
from functools import reduce
from retrieve_hadoop import retrieve_from_hadoop
import os

# Function to read tweets from the JSON file
def read_tweets(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Mapper function to extract (key, value) pairs
def map_function(tweets):
    intermediate_data = []
    for tweet in tweets:
        country_code = tweet['country']
        post_date = tweet['event_date']
        key = (country_code, post_date)
        value = tweet
        intermediate_data.append((key, value))
    return intermediate_data

# Reducer function to group values by key
def reduce_function(intermediate_data):
    grouped_data = defaultdict(list)
    for key, value in intermediate_data:
        grouped_data[key].append(value)
    return grouped_data

# Main execution to simulate the MapReduce process
def map_reduce(file_path):
    # Step 1: Read tweets from the JSON file
    tweets = read_tweets(file_path)

    # Step 2: Apply the map function to get intermediate data
    intermediate_data = map_function(tweets)

    # Step 3: Apply the reduce function to group data by key
    grouped_data = reduce_function(intermediate_data)

    return grouped_data

# Output grouped tweets to JSON with stringified keys
def save_grouped_data_to_json(grouped_data, output_file_path):
    # Convert keys from tuples to strings
    stringified_data = {str(key): value for key, value in grouped_data.items()}
   
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(stringified_data, file, ensure_ascii=False, indent=4)

def get_local_json_file_path(local_output_path):
    for filename in os.listdir(local_output_path):
        if filename.endswith(".json"):
            return os.path.join(local_output_path, filename)
    return None

def make_valid_json_file(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
   
    json_content = '[' + ','.join(line.strip() for line in lines) + ']'

    try:
        json_data = json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        exit(1)

# Write the valid JSON back to the file
    with open(input_file, 'w') as file:
        json.dump(json_data, file, indent=4)


# Run the MapReduce process
if __name__ == "__main__":
        
    local_output_path = "/home/maverick/Documents/MTech_sem3/SDE/Final_project/landslide_analysis/data/reddit_data_landslide"
    output_dir = retrieve_from_hadoop(local_output_path)
    print(f"Data appended to: {output_dir}")

    json_file_path = get_local_json_file_path(local_output_path)

    make_valid_json_file(json_file_path)
   
    # input_file_path = '/home/rudy/HTwitt/tweets_output.json'  # Path to the input JSON file
    output_file_path = '/home/maverick/Documents/MTech_sem3/SDE/Final_project/landslide_analysis/data/reddit_data_landslide/grouped_tweets_output.json'  # Path to the output JSON file

    grouped_tweets = map_reduce(json_file_path)
    save_grouped_data_to_json(grouped_tweets, output_file_path)

    print(f"Grouped data saved to {output_file_path}")


