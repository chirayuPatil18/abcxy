### Design RNN or it’s variant including LSTM or GRU a. Select a suitable time series dataset. Example :- predict sentiments based on product reviews.
# Import Libraries
import numpy as np

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load Dataset
vocab_size = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

# Data Preprocessing
max_length = 200

X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Define LSTM Model
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),

    LSTM(128),

    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64
)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

# Predict One Review
prediction = model.predict(np.array([X_test[0]]))

print("Prediction Score:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Positive Review")
else:
    print("Negative Review")