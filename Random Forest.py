!pip install nltk

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
nltk.download('punkt')

data = pd.read_csv('/content/Twitter_Data.csv')
data

# Assuming you have a DataFrame named 'data'
import pandas as pd

# Remove rows with missing values from the entire DataFrame
data.dropna(inplace=True)

# To remove missing values from a specific column, for example 'label'
data.dropna(subset=['clean_text'], inplace=True)

# Assuming your dataset has 'text' and 'label' columns
text_data = data['clean_text']
labels = data['category']

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    # Join the tokens back into a string
    return ' '.join(tokens)

# Apply the preprocessing function to the text data
text_data = text_data.apply(preprocess_text)

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer with a maximum of 5000 features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the preprocessed text data to create the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)
