import json

# Class to represent each tweet
class Tweet:
    def __init__(self, event_title, event_date, location, description, fatalities, trigger, country):
        self.event_title = event_title
        self.event_date = event_date
        self.location = location
        self.description = description
        self.fatalities = fatalities
        self.trigger = trigger
        self.country = country

    def __str__(self):
        return (f"Tweet{{event_title='{self.event_title}', "
                f"event_date='{self.event_date}', "
                f"location='{self.location}', "
                f"description='{self.description}', "
                f"fatalities='{self.fatalities}', "
                f"trigger='{self.trigger}', "
                f"country='{self.country}'}}")

# Function to read and parse the JSON file into Tweet objects
def read_tweets_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Load JSON data from the file
            data = json.load(file)

            # Deserialize JSON data into a list of Tweet objects
            tweets = [
                Tweet(
                    tweet['event_title'],
                    tweet['event_date'],
                    tweet['location'],
                    tweet['description'],
                    tweet['fatalities'],
                    tweet['trigger'],
                    tweet['country']
                ) for tweet in data
            ]

            # Print each tweet (or process as needed)
            for tweet in tweets:
                print(tweet)

    except FileNotFoundError:
        print("Error: JSON file not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON.")
    except Exception as e:
        print(f"Error reading the JSON file: {e}")

# Main execution
file_path = '/home/rudy/HTwitt/tweets_output.json'  # Path to the JSON file
read_tweets_from_json(file_path)