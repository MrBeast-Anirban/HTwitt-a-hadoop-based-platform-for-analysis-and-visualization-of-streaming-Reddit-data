import json
import re
import nltk
import time
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import numpy as np

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def preprocess_tweet(tweet):
    tweet_text = tweet['description']  # Extract tweet-like text from the 'description' field
    tweet_text = re.sub(r"[^a-zA-Z\s]", "", tweet_text)  # Remove non-alphabetic characters
    tweet_text = tweet_text.lower()  # Convert to lowercase
    tokens = word_tokenize(tweet_text)  # Tokenize tweet text
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens)  # Join tokens back into a string

# Load grouped tweets from the saved JSON file
def load_grouped_tweets(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        grouped_tweets = json.load(file)
    return grouped_tweets

# Feature extraction functions
def get_ngrams(tweets, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    ngrams = vectorizer.fit_transform(tweets)
    return ngrams, vectorizer

def get_tfidf_features(tweets):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)
    return tfidf_matrix, tfidf_vectorizer

# Load and preprocess grouped tweets
grouped_tweets_file_path = "C:\Users\AVA\Documents\MTech-sem3\SDE\SDE major project\landslide_analysis\landslide_analysis\src\grouped_reddit_output.json"
grouped_tweets = load_grouped_tweets(grouped_tweets_file_path)

all_tweets = []
for country_date, tweets in grouped_tweets.items():
    for tweet in tweets:
        preprocessed_tweet = preprocess_tweet(tweet)
        all_tweets.append(preprocessed_tweet)

# Dummy labels for classification (replace with actual labels)
labels = [1 if 'landslide' in tweet else 0 for tweet in all_tweets]  # Replace this with your actual label logic

# Hyperparameter Grid
ngram_configs = [1, 2, 3, 4, 5]  # Unigrams, Bigrams, Trigrams, 4-Grams, 5-Grams
split_ratios = [0.2, 0.25, 0.3]  # Split ratios (including 20%)
alphas = [0.2, 0.4, 0.6, 0.8, 1.0]  # Naive Bayes smoothing parameter
vectorizer_choices = ['count', 'tfidf']  # Choice of vectorizer

# Results storage
best_results = {ngram_range: {} for ngram_range in ngram_configs}

# Loop through different hyperparameter configurations
for ngram_range in ngram_configs:
    for split_ratio in split_ratios:
        for alpha in alphas:
            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(all_tweets, labels, test_size=split_ratio, random_state=42)

            # Feature extraction based on vectorizer choice
            vectorizer = CountVectorizer(ngram_range=(ngram_range, ngram_range)) if vectorizer_choices[0] == 'count' else TfidfVectorizer()
            ngrams = vectorizer.fit_transform(X_train)

            # Train the Naïve Bayes classifier
            naive_bayes_classifier = MultinomialNB(alpha=alpha)

            # Time the training process
            start_time = time.time()
            naive_bayes_classifier.fit(ngrams, y_train)
            train_time = time.time() - start_time

            # Predict and evaluate the model on the test set
            X_test_ngrams = vectorizer.transform(X_test)
            start_time = time.time()
            y_pred = naive_bayes_classifier.predict(X_test_ngrams)
            test_time = time.time() - start_time

            # Compute evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            std = np.std([accuracy, f1, precision, recall, kappa])

            # Print Accuracy for this configuration
            print(f"Accuracy for {ngram_range}-gram model, split {split_ratio*100}%, alpha={alpha}: {accuracy:.4f}")

            # Store the best result for each ngram_range and split_ratio
            if split_ratio not in best_results[ngram_range] or best_results[ngram_range][split_ratio]['accuracy'] < accuracy:
                best_results[ngram_range][split_ratio] = {
                    'alpha': alpha,
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'std': std,
                    'kappa': kappa
                }

# Visualization of Results for each n-gram model
for ngram_range in ngram_configs:
    splits = [split_ratio * 100 for split_ratio in split_ratios]
    accuracy_values = [best_results[ngram_range][split_ratio]['accuracy'] for split_ratio in split_ratios]
    f1_values = [best_results[ngram_range][split_ratio]['f1'] for split_ratio in split_ratios]
    precision_values = [best_results[ngram_range][split_ratio]['precision'] for split_ratio in split_ratios]
    recall_values = [best_results[ngram_range][split_ratio]['recall'] for split_ratio in split_ratios]
    std_values = [best_results[ngram_range][split_ratio]['std'] for split_ratio in split_ratios]
    kappa_values = [best_results[ngram_range][split_ratio]['kappa'] for split_ratio in split_ratios]

    # Bar width and offset settings
    bar_width = 0.15
    bar_positions = np.arange(len(splits))

    # Plotting all metrics as bar plots for the current ngram_range
    plt.figure(figsize=(16, 8))
    
    plt.bar(bar_positions, precision_values, width=bar_width, label="Precision", color='blue')
    plt.bar(bar_positions + bar_width, recall_values, width=bar_width, label="Recall", color='green')
    plt.bar(bar_positions + 2 * bar_width, f1_values, width=bar_width, label="F1 Score", color='red')
    plt.bar(bar_positions + 3 * bar_width, accuracy_values, width=bar_width, label="Accuracy", color='cyan')
    plt.bar(bar_positions + 4 * bar_width, kappa_values, width=bar_width, label="Kappa", color='orange')
    plt.bar(bar_positions + 5 * bar_width, std_values, width=bar_width, label="Std", color='purple')
    
    # Customize plot
    plt.title(f"Performance Metrics for {ngram_range}-gram Model")
    plt.xlabel("Split Percentage")
    plt.ylabel("Scores")
    plt.xticks(bar_positions + 2.5 * bar_width, [int(split) for split in splits])  # Centering x-tick labels
    plt.ylim([0, 1])  # All scores are normalized between 0 and 1
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')

    # Display the plot
    plt.tight_layout()
    plt.show()

# Comparative Visualization of Accuracies for all n-gram models
plt.figure(figsize=(18, 10))

# Bar width and offset settings
bar_width = 0.15
ngram_labels = [f"{ngram_range}-gram" for ngram_range in ngram_configs]

# Loop through each n-gram range and plot accuracies for different split ratios
for i, ngram_range in enumerate(ngram_configs):
    splits = [split_ratio * 100 for split_ratio in split_ratios]
    accuracy_values = [best_results[ngram_range][split_ratio]['accuracy'] for split_ratio in split_ratios]
    
    # Adjust position for each n-gram range's bars
    bar_positions = np.arange(len(splits)) + i * bar_width
    plt.bar(bar_positions, accuracy_values, width=bar_width, label=f"{ngram_range}-gram")

# Customize plot
plt.title("Comparison of Accuracies Across N-gram Models and Split Percentages")
plt.xlabel("Split Percentage")
plt.ylabel("Accuracy")
plt.xticks(np.arange(len(splits)) + (len(ngram_configs) / 2 - 0.5) * bar_width, [int(split) for split in splits])
plt.ylim([0, 1])
plt.legend(title="N-gram Models", loc='lower right')
plt.tight_layout()
plt.show()

