import json
import re
import nltk
import time
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

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

# Load and preprocess grouped tweets
grouped_tweets_file_path = "grouped_tweets_output.json"
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
split_ratios = [0.2, 0.25, 0.5, 0.75, 0.8, 0.9]  # Split ratios (including 20%)
alphas = [0.2, 0.4, 0.6, 0.8, 1.0]  # Naive Bayes smoothing parameter
vectorizer_choices = ['count', 'tfidf']  # Choice of vectorizer

# Tokenizer for text data
tokenizer = Tokenizer()

# Results storage
results = {ngram_range: [] for ngram_range in ngram_configs}

# Loop through different hyperparameter configurations
for ngram_range in ngram_configs:
    for split_ratio in split_ratios:
        for alpha in alphas:
            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(all_tweets, labels, test_size=split_ratio, random_state=42)

            # Update the tokenizer to handle n-grams
            tokenizer.fit_on_texts(X_train)

            # Convert text to sequences (convert text to sequences of integers)
            X_train_seq = tokenizer.texts_to_sequences(X_train)
            X_test_seq = tokenizer.texts_to_sequences(X_test)

            # Pad sequences to ensure equal length (max_length can be adjusted)
            max_length = max([len(seq) for seq in X_train_seq])  # Adjust this as needed
            X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
            X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

            # Define the deep learning model (MLP)
            model = Sequential()
            model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_length))
            model.add(GlobalAveragePooling1D())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))

            # Compile the model
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Time the training process
            start_time = time.time()
            model.fit(X_train_pad, np.array(y_train), epochs=5, batch_size=64, verbose=0)
            train_time = time.time() - start_time

            # Predict and evaluate the model on the test set
            y_pred = (model.predict(X_test_pad) > 0.5).astype(int)

            # Compute evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            std = np.std([accuracy, f1, precision, recall, kappa])

            # Print Accuracy for this configuration
            print(f"Accuracy for {ngram_range}-gram model, split {split_ratio*100}%, alpha={alpha}: {accuracy:.4f}")

            # Store results for this n-gram model, split ratio, and alpha
            results[ngram_range].append({
                'split_ratio': split_ratio,
                'alpha': alpha,
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'std': std,
                'kappa': kappa
            })

# Visualization of Results for each n-gram model
for ngram_range in ngram_configs:
    splits = [result['split_ratio'] * 100 for result in results[ngram_range]]
    accuracy_values = [result['accuracy'] for result in results[ngram_range]]
    f1_values = [result['f1'] for result in results[ngram_range]]
    precision_values = [result['precision'] for result in results[ngram_range]]
    recall_values = [result['recall'] for result in results[ngram_range]]
    std_values = [result['std'] for result in results[ngram_range]]
    kappa_values = [result['kappa'] for result in results[ngram_range]]

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
    splits = [result['split_ratio'] * 100 for result in results[ngram_range]]
    accuracy_values = [result['accuracy'] for result in results[ngram_range]]
    plt.bar(np.arange(len(splits)) + i * bar_width, accuracy_values, width=bar_width, label=f"{ngram_range}-gram")

# Customize plot
plt.title("Comparison of Accuracies for Different N-Grams")
plt.xlabel("Split Percentage")
plt.ylabel("Accuracy")
plt.xticks(np.arange(len(splits)) + bar_width * (len(ngram_configs) / 2), [int(split) for split in splits])
plt.legend(title="N-Gram Range")
plt.grid(True, linestyle='--', alpha=0.5)

# Display the plot
plt.tight_layout()
plt.show()
