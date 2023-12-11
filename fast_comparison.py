import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv('combined.csv')

# Drop rows with missing values in 'author' or 'content' columns
data = data.dropna(subset=['author', 'content'])

# Preprocess the data
X = data['content']
y = data['label']

# Convert 'Elon' and 'Other' labels to binary values (0 and 1)
y = y.map({'Elon': 1, 'Other': 0})

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF vectorizer
# You can adjust max_features as needed
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize classifiers
classifiers = {
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression()
}

# Train and evaluate each classifier
for clf_name, clf in classifiers.items():
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print results
    print(f"Classifier: {clf_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")
    print("-------------------------------------------------------")
