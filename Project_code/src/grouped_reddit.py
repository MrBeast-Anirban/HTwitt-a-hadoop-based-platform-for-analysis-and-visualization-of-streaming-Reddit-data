from collections import defaultdict
import json

# Function to read posts from the JSON file
def read_reddit_posts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Mapper function to extract (key, value) pairs based on available fields
def map_function(posts):
    intermediate_data = []
    for post in posts:
        # Prioritize grouping by country and event_date if available, else use the trigger field
        country_code = post.get('country')
        post_date = post.get('event_date')
        trigger = post.get('trigger')

        if country_code and post_date:
            key = (country_code, post_date)
        elif trigger:
            key = (trigger,)
        else:
            # Skip posts that don't have either grouping criteria
            continue

        value = post
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
    # Step 1: Read posts from the JSON file
    posts = read_reddit_posts(file_path)

    # Step 2: Apply the map function to get intermediate data
    intermediate_data = map_function(posts)

    # Step 3: Apply the reduce function to group data by key
    grouped_data = reduce_function(intermediate_data)

    return grouped_data

# Output grouped posts to JSON with stringified keys
def save_grouped_data_to_json(grouped_data, output_file_path):
    # Convert keys from tuples to strings
    stringified_data = {str(key): value for key, value in grouped_data.items()}
   
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(stringified_data, file, ensure_ascii=False, indent=4)

# Run the MapReduce process
if __name__ == "__main__":
    input_file_path = '/home/rudy/HTwitt/reddit_data.json'  # Path to the input JSON file
    output_file_path = '/home/rudy/HTwitt/grouped_reddit_output.json'  # Path to the output JSON file

    grouped_posts = map_reduce(input_file_path)
    save_grouped_data_to_json(grouped_posts, output_file_path)

    print(f"Grouped data saved to {output_file_path}")
