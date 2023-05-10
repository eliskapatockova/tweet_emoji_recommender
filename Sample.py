import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("tweets_emojis.csv")

# Extract features from the text using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
X = vectorizer.fit_transform(df['Tweet'])

# Encode the emoji labels as integers
emoji_labels = df['Emoji'].astype('category').cat.codes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, emoji_labels, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Train a Neural Network classifier
nn_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn_classifier.fit(X_train, y_train)

# Make predictions on the test set with Random Forest classifier
rf_y_pred = rf_classifier.predict(X_test)

# Convert the predicted labels back to emoji labels
predicted_emojis = pd.Series(rf_y_pred).astype('category').cat.categories
rf_y_pred_emojis = pd.Series(rf_y_pred).astype('category').cat.rename_categories(predicted_emojis)
y_test_emojis = pd.Series(y_test).astype('category').cat.rename_categories(predicted_emojis)

# Evaluate the performance of the Random Forest classifier
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("Random Forest Accuracy: ", rf_accuracy)

# Make predictions on the test set with Neural Network classifier
nn_y_pred = nn_classifier.predict(X_test)

# Convert the predicted labels back to emoji labels
predicted_emojis = pd.Series(nn_y_pred).astype('category').cat.categories
nn_y_pred_emojis = pd.Series(nn_y_pred).astype('category').cat.rename_categories(predicted_emojis)
y_test_emojis = pd.Series(y_test).astype('category').cat.rename_categories(predicted_emojis)

# Evaluate the performance of the Neural Network classifier
nn_accuracy = accuracy_score(y_test, nn_y_pred)
print("Neural Network Accuracy: ", nn_accuracy)
