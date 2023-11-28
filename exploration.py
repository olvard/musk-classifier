from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

file_path1 = "data/modified_file_musk.csv"
file_path2 = "data/modified_file_others.csv"
elon = pd.read_csv(file_path1)
other = pd.read_csv(file_path2)

# Swapping columns 1 and 2 (indexing starts from 0)
elon = elon[['user_name', 'text', 'date', 'favorites']]

# Convert 'date' column in elon_tweets to match the format of 'date_time' in the other dataset
elon['date'] = pd.to_datetime(elon['date']).dt.strftime('%d/%m/%Y %H:%M')

# Rename columns in 'elon_tweets' DataFrame
elon = elon.rename(columns={
    'user_name': 'author',  # Change 'user_name' to 'author'
    'text': 'content',      # Change 'text' to 'content'
    'date': 'date_time',    # Change 'date' to 'date_time'
    'favorites': 'number_of_likes'  # Change 'favorites' to 'number_of_likes'
})

combined_tweets = pd.concat([other, elon])

# Assign labels to tweets (assuming 'Elon Musk' and 'Other Author')
combined_tweets['label'] = combined_tweets['author'].apply(
    lambda x: 'Elon Musk' if x == 'Elon Musk' else 'Other Author')

# Split data into features (tweets) and labels
X = combined_tweets['content']
y = combined_tweets['label']

# Initialize the TF-IDF vectorizer
# You can adjust max_features as needed
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Transform the text data into TF-IDF features
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42)

# Initialize a classifier (SVM in this case)
classifier = SVC(kernel='linear')

# Train the classifier
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate classification report
print(classification_report(y_test, y_pred))
