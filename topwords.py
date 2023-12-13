import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the data
combined_tweets = pd.read_csv('combined.csv')

# Filter the dataset to include only entries labeled as 'Elon'
elon_tweets = combined_tweets[combined_tweets['label'] == 'Elon Musk']

print(elon_tweets.head())

# Fill missing values in the 'content' column and convert non-string types to strings
elon_tweets['content'] = elon_tweets['content'].fillna('').astype(str)

# Split data into features (tweets) and labels for Elon tweets
X_elon = elon_tweets['content']
y_elon = elon_tweets['label']

# Initialize the TF-IDF vectorizer for Elon's tweets
tfidf_vectorizer_elon = TfidfVectorizer(max_features=3500)

# Transform the text data into TF-IDF features for Elon's tweets
X_tfidf_elon = tfidf_vectorizer_elon.fit_transform(X_elon)

# Get the feature names (words) from the TF-IDF vectorizer for Elon's tweets
feature_names_elon = tfidf_vectorizer_elon.get_feature_names_out()

# Sum up the occurrences of each word across all Elon's documents
word_frequencies_elon = X_tfidf_elon.sum(axis=0)

# Create a dictionary to pair each word with its frequency for Elon's tweets
word_frequency_dict_elon = {}
for word, freq in zip(feature_names_elon, word_frequencies_elon.tolist()[0]):
    word_frequency_dict_elon[word] = freq

# Sort the words by their frequencies in descending order for Elon's tweets
sorted_word_frequencies_elon = sorted(
    word_frequency_dict_elon.items(), key=lambda x: x[1], reverse=True)

# Display the top five most used words for Elon's tweets
top_five_words_elon = sorted_word_frequencies_elon[:20]
for word, frequency in top_five_words_elon:
    print(f"Word: {word}, Frequency: {frequency}")
