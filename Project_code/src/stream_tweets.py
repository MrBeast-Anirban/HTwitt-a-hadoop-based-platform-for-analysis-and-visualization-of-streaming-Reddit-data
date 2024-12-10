import csv
import json

# Function to read and extract information from the CSV
def extract_landslide_data(csv_file_path):
    tweets = []
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Extract relevant columns for tweet content
            event_title = row.get("event_title", "").strip()
            event_date = row.get("event_date", "").strip()
            location = row.get("location_description", "").strip()
            description = row.get("event_description", "").strip()
            fatalities = row.get("fatality_count", "unknown").strip()
            trigger = row.get("landslide_trigger", "").strip()
            country = row.get("country_name", "").strip()

            # Format the extracted data into a structured JSON object
            tweet = {
                "event_title": event_title,
                "event_date": event_date,
                "location": location,
                "description": description,
                "fatalities": fatalities,
                "trigger": trigger,
                "country": country
            }
           
            # Add the JSON object to the list
            tweets.append(tweet)
   
    return tweets

# Save extracted tweets to a JSON file
def save_tweets_to_json(tweets, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(tweets, f, ensure_ascii=False, indent=4)

# Main execution
csv_file_path = '/home/rudy/HTwitt/Global_Landslide_Catalog_Export.csv'  # Path to your CSV file
output_file_path = '/home/rudy/HTwitt/tweets_output.json'  # Output file for storing tweets in JSON format

# Extract data and save to JSON file
tweets = extract_landslide_data(csv_file_path)
save_tweets_to_json(tweets, output_file_path)

print(f"Extracted {len(tweets)} tweets and saved them to {output_file_path}")