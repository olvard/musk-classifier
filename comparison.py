from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB 

combined_tweets = pd.read_csv('combined.csv')

print(combined_tweets.head())

# Fill missing values in the 'content' column and convert non-string types to strings
combined_tweets['content'] = combined_tweets['content'].fillna('').astype(str)

# Split data into features (tweets) and labels
X = combined_tweets['content']
y = combined_tweets['label']

# Initialize the TF-IDF vectorizer
# You can adjust max_features as needed
tfidf_vectorizer = TfidfVectorizer(max_features=1500)

# Transform the text data into TF-IDF features
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=20)


# --------------------classifiers-------------------

# Initialize classifiers
svm_classifier = SVC(kernel='linear', random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
nb_classifier = MultinomialNB()  # Multinomial Naive Bayes

# Train SVM classifier
svm_classifier.fit(X_train, y_train)

# Train RandomForest classifier
rf_classifier.fit(X_train, y_train)

# Train KNeighbors classifier
knn_classifier.fit(X_train, y_train)

# Train Naive Bayes classifier
nb_classifier.fit(X_train, y_train)

# Make predictions
svm_y_pred = svm_classifier.predict(X_test)
rf_y_pred = rf_classifier.predict(X_test)
knn_y_pred = knn_classifier.predict(X_test)
nb_y_pred = nb_classifier.predict(X_test) 

# Evaluate and print accuracy for each classifier
svm_accuracy = accuracy_score(y_test, svm_y_pred)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
knn_accuracy = accuracy_score(y_test, knn_y_pred)
nb_accuracy = accuracy_score(y_test, nb_y_pred) 

# Print accuracies
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print(f"RandomForest Accuracy: {rf_accuracy:.2f}")
print(f"KNN Accuracy: {knn_accuracy:.2f}")
print(f"Naive Bayes Accuracy: {nb_accuracy:.2f}") 

# Generate classification report for SVM
print("SVM Classification Report:")
print(classification_report(y_test, svm_y_pred))

# Generate classification report for RandomForest
print("RandomForest Classification Report:")
print(classification_report(y_test, rf_y_pred))

# Generate classification report for KNN
print("KNN Classification Report:")
print(classification_report(y_test, knn_y_pred))

# Generate classification report for Naive Bayes
print("Naive Bayes Classification Report:")
print(classification_report(y_test, nb_y_pred))

# Visualize confusion matrix for SVM
conf_matrix_svm = confusion_matrix(y_test, svm_y_pred)
conf_matrix_knn = confusion_matrix(y_test, knn_y_pred)
conf_matrix_rf = confusion_matrix(y_test, rf_y_pred)
conf_matrix_nb = confusion_matrix(y_test, nb_y_pred)

# Creating a figure with 1 row and 3 columns for subplots
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

# Plotting SVM Confusion Matrix
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues',
            xticklabels=svm_classifier.classes_, yticklabels=svm_classifier.classes_,
            ax=axes[0])  # Plotting in the first subplot (index 0)
axes[0].set_xlabel('Predicted Labels')
axes[0].set_ylabel('True Labels')
axes[0].set_title('SVM Confusion Matrix')

# Plotting RandomForest Confusion Matrix
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=rf_classifier.classes_, yticklabels=rf_classifier.classes_,
            ax=axes[1])  # Plotting in the second subplot (index 1)
axes[1].set_xlabel('Predicted Labels')
axes[1].set_ylabel('True Labels')
axes[1].set_title('RandomForest Confusion Matrix')

# Plotting KNN Confusion Matrix
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues',
            xticklabels=knn_classifier.classes_, yticklabels=knn_classifier.classes_,
            ax=axes[2])  # Plotting in the third subplot (index 2)
axes[2].set_xlabel('Predicted Labels')
axes[2].set_ylabel('True Labels')
axes[2].set_title('KNN Confusion Matrix')

# Plotting Naive Bayes Confusion Matrix
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues',
            xticklabels=nb_classifier.classes_, yticklabels=nb_classifier.classes_,
            ax=axes[3])  # Plotting in the fourth subplot (bottom-right)
axes[3].set_xlabel('Predicted Labels')
axes[3].set_ylabel('True Labels')
axes[3].set_title('Naive Bayes Confusion Matrix')

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()
