import praw
from reddit_collector import collect_landslide_data
from hadoop_processor import init_hadoop_processing
import json

def main():
    # Initialize Reddit instance
    reddit = praw.Reddit(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="script:datacollector:v1.0 (by /u/YOUR_REDDIT_USERNAME)",
        username="YOUR_REDDIT_USERNAME",
        password="YOUR_REDDIT_PASSWORD"
    )
    
    # Step 1: Collect Reddit data
    # Uncomment this line to collect data from the Reddit API
    json_file = collect_landslide_data(reddit)  
    
    json_file  = "file:///home/maverick/Documents/MTech_sem3/SDE/Final_project/landslide_analysis/src/reddit_data_landslide/new_landslide_events_20241114_183847.json"
    
    
    my_file = "/home/maverick/Documents/MTech_sem3/SDE/Final_project/landslide_analysis/src/reddit_data_landslide/new_landslide_events_20241114_183847.json"
    
    #Debugging 
    # with open(my_file, 'r', encoding='utf-8') as file:
    #     data = json.load(file)
    # print(data[:5]) 
   
    #Invoking he function to store the data in hadoop
    if json_file:
        # Step 2: Process with Hadoop
        init_hadoop_processing(json_file)
    else:
        print("No data collected to process")

if __name__ == "__main__":
    main()