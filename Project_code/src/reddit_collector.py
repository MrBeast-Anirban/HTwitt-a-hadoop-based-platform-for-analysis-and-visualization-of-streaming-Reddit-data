import praw
import json
import time
from datetime import datetime
import os
import re

def parse_date_from_title(title):
    """Try to extract date from title or return current date as default"""
    # Add more date patterns if needed
    date_patterns = [
        r'(\d{1,2}/\d{1,2}/\d{4})',
        r'(\d{4}-\d{1,2}-\d{1,2})',
        r'(\d{1,2}-\d{1,2}-\d{4})'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, title)
        if match:
            try:
                date_str = match.group(1)
                date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                return date_obj.strftime('%m/%d/%Y %I:%M:%S %p')
            except:
                continue
    
    # Return current date if no date found
    return datetime.now().strftime('%m/%d/%Y %I:%M:%S %p')

def extract_location(title, text):
    """Extract location information from title or text"""
    # This is a simple implementation - you might want to use NLP for better results
    location_keywords = ["in", "at", "near", "around"]
    
    for keyword in location_keywords:
        pattern = f"{keyword} ([^,.]+)"
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return "Location not specified"

def extract_fatalities(text):
    """Extract fatality numbers from text"""
    patterns = [
        r'(\d+)\s*killed',
        r'(\d+)\s*dead',
        r'(\d+)\s*fatalities',
        r'death toll\s*[:-]?\s*(\d+)',
        r'(\d+)\s*people\s*died'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return "0"

def identify_trigger(text):
    """Identify the trigger of the landslide"""
    triggers = {
        'rain': ['rain', 'rainfall', 'heavy rain', 'downpour'],
        'earthquake': ['earthquake', 'seismic', 'tremor'],
        'construction': ['construction', 'building', 'excavation'],
        'mining': ['mining', 'mine'],
        'erosion': ['erosion', 'eroded']
    }
    
    text_lower = text.lower()
    for trigger_type, keywords in triggers.items():
        for keyword in keywords:
            if keyword in text_lower:
                return trigger_type
    
    return "unknown"

def collect_landslide_data(reddit):
    output_dir = "reddit_data_landslide"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define subreddits and keywords
    subreddits = [
        'NaturalDisasters',
        'geology', 
        'weather', 
        'climate',
        'GeologicalDisasters',
        'worldnews',
        'science',
        'naturaldisaster',
        'india',  # Country-specific subreddits for regional coverage
        'china',
        'japan',
        'indonesia',
        'philippines',
        'disaster',
        'ClimateChange'
    ]
    keywords = [
        # Direct landslide terms
        'landslide',
        'mudslide',
        'landfall',
        'landslip',
        'soil sliding',
        'rockslide',
        
        # Related terms
        'mountain collapse',
        'hillside collapse',
        'slope collapse',
        'soil collapse',
    
        # Compound searches
        'hill AND collapse',
        'mountain AND disaster',
        'soil AND disaster',
        
    ]
    time_filters = ['all', 'year','month', 'week']
    
    landslide_data = {}
    
    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            print(f"Collecting from r/{subreddit_name}...")
            
            for keyword in keywords:
                print(f"  Searching for keyword: {keyword}")
                
                # Search across different time periods
                for time_filter in time_filters:
                    try:
                        # Use different sort methods to get more diverse results
                        for sort in ['relevance', 'top', 'new']:
                            submissions = subreddit.search(
                                keyword, 
                                limit=100,
                                time_filter=time_filter,
                                sort=sort
                            )
                            
                            for submission in submissions:
                                # Skip if it's a duplicate
                                if submission.id in [event['id'] for events in landslide_data.values() 
                                                   for event in events if 'id' in event]:
                                    continue
                                
                                # Combine title and selftext for better information extraction
                                full_text = f"{submission.title}\n{submission.selftext}"
                                
                                # Try to get comments for additional information
                                try:
                                    submission.comments.replace_more(limit=0)
                                    comments_text = "\n".join([comment.body for comment in submission.comments.list()[:5]])
                                    full_text += "\n" + comments_text
                                except:
                                    pass
                                
                                # Extract date
                                event_date = parse_date_from_title(submission.title)
                                
                                # Extract location and country
                                location = extract_location(submission.title, submission.selftext)
                                country = "Country not specified"  # You might want to use a geo-coding service here
                                
                                # Create event entry
                                event_data = {
                                    "id": submission.id,
                                    "event_title": submission.title,
                                    "event_date": event_date,
                                    "location": location,
                                    "description": submission.selftext[:500] if submission.selftext else submission.title,
                                    "fatalities": extract_fatalities(full_text),
                                    "trigger": identify_trigger(full_text),
                                    "country": country,
                                    "url": f"https://reddit.com{submission.permalink}",
                                    "score": submission.score,
                                    "num_comments": submission.num_comments,
                                    "matched_keyword": keyword,
                                    "subreddit": subreddit_name
                                }
                                
                                # Use country and date as key
                                key = f"({country}, {event_date})"
                                if key not in landslide_data:
                                    landslide_data[key] = []
                                landslide_data[key].append(event_data)
                                
                                print(f"    Processed: {submission.title[:50]}...")
                                time.sleep(2)  # Respect rate limits
                    
                    except Exception as e:
                        print(f"Error with time filter {time_filter}: {str(e)}")
                        continue
            
        except Exception as e:
            print(f"Error collecting from r/{subreddit_name}: {str(e)}")
    
    # Save to file
    if landslide_data:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{output_dir}/landslide_events_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(landslide_data, f, ensure_ascii=False, indent=4)
        print(f"Saved landslide data to {filename}")
        print(f"Total events collected: {sum(len(events) for events in landslide_data.values())}")
        return filename
    else:
        print("No landslide data collected")
        return None
