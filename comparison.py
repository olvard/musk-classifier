from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


# Load data from CSV
data = pd.read_csv('combined.csv')

# Assuming 'feature1' and 'feature2' are your feature columns and 'target' is the target column
X = data[['content']].values
y = data['label'].values

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(random_state=42),
]

figure = plt.figure(figsize=(27, 9))
i = 1

# Initialize the TF-IDF vectorizer
# You can adjust max_features as needed
tfidf_vectorizer = TfidfVectorizer(max_features=1500)

# Transform the text data into TF-IDF features
X_tfidf = tfidf_vectorizer.fit_transform(X)

# preprocess dataset, split into training and test part
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=40)

# Apply PCA to reduce the TF-IDF features to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf.toarray())

x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5


cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train,
            cmap=cm_bright, edgecolors="k", label='Training Data')

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test,
            cmap=cm_bright, alpha=0.6, edgecolors="k", label='Testing Data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.legend()

# iterate over classifiers
for name, clf in zip(names, classifiers):
    plt.figure(figsize=(6, 4))
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # Create scatter plot for training data
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train,
                cmap=cm_bright, edgecolors="k", label='Training Data')

    # Create scatter plot for testing data
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test,
                cmap=cm_bright, alpha=0.6, edgecolors="k", label='Testing Data')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.title(f"{name} - Accuracy: {score:.2f}")
    plt.legend()

plt.tight_layout()
plt.show()

# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(1, len(classifiers) + 1, i + 1)

    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # DecisionBoundaryDisplay is not compatible with all classifiers, so let's use scatter plots instead
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train,
               cmap=cm_bright, edgecolors="k")
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test,
               cmap=cm_bright, alpha=0.6, edgecolors="k")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(
        x_max - 0.3,
        y_min + 0.3,
        ("%.2f" % score).lstrip("0"),
        size=15,
        horizontalalignment="right",
    )
    i += 1

plt.tight_layout()
plt.show()
