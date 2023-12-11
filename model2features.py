from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import hstack

# Load the data
combined_tweets = pd.read_csv('combined.csv')

print(combined_tweets.head())

# Fill missing values in the 'content' column and convert non-string types to strings
combined_tweets['content'] = combined_tweets['content'].fillna('').astype(str)

# Split data into features (tweets) and labels
X_text = combined_tweets['content']
X_likes = combined_tweets['number_of_likes'].values.reshape(-1, 1)
y = combined_tweets['label']

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1500)

# Transform the text data into TF-IDF features
X_tfidf = tfidf_vectorizer.fit_transform(X_text)

# Combine 'number_of_likes' with TF-IDF features
X_combined = hstack((X_tfidf, X_likes))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42)

# Initialize a classifier (SVM in this case)
classifier = SVC(kernel='linear', random_state=42)

# Train the classifier on combined features
classifier.fit(X_train, y_train)

# Predict on the test set with combined features
y_pred = classifier.predict(X_test)

# Calculate accuracy with combined features
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with combined features: {accuracy:.2f}")

# Generate classification report with combined features
print(classification_report(y_test, y_pred))

# Visualize confusion matrix with combined features
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=classifier.classes_, yticklabels=classifier.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix with Combined Features')
plt.show()
