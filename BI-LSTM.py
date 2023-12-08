!pip install tensorflow
!pip install nltk

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
nltk.download('punkt')

data = pd.read_csv('/content/Twitter_Data.csv', dtype={'text': str})

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

max_words = 10000  # Maximum number of words to keep in the vocabulary
max_sequence_length = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data['clean_text'])
sequences = tokenizer.texts_to_sequences(data['clean_text'])
X = pad_sequences(sequences, maxlen=max_sequence_length)
y = data['category']

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer with a maximum of 5000 features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the preprocessed text data to create the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)

model = Sequential()
model.add(Embedding(max_words, 32, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')
