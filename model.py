from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB


# Load the data
combined_tweets = pd.read_csv('combined.csv')

print(combined_tweets.head())

# Fill missing values in the 'content' column
combined_tweets['content'] = combined_tweets['content'].fillna('').astype(str)

# Split data into features (tweets) and labels
X = combined_tweets['content']
y = combined_tweets['label']

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=3500)

X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42)

# Grid Search
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
}

classifier = MultinomialNB()
grid_search = GridSearchCV(classifier, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_classifier = MultinomialNB(alpha=best_params['alpha'])

best_classifier.fit(X_train, y_train)

y_pred = best_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))

# confusion matrix after tuning
conf_matrix_tuned = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_tuned, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_classifier.classes_, yticklabels=best_classifier.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix after tuning')
plt.show()
