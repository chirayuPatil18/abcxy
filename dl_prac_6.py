### RNN-based Sentiment Analysis on Network Graph 
# install lib - !pip install networkx
# Import Libraries
import numpy as np
import networkx as nx

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create Graph Data
G = nx.Graph()

reviews = [
    ("good product", 1),
    ("excellent quality", 1),
    ("bad experience", 0),
    ("poor service", 0)
]

# Add Nodes
for i, (text, label) in enumerate(reviews):
    G.add_node(i, text=text, sentiment=label)

# Add Edges
G.add_edges_from([(0,1), (2,3)])

# Extract Text and Labels
texts = []
labels = []

for node in G.nodes(data=True):
    texts.append(node[1]['text'])
    labels.append(node[1]['sentiment'])

# Text Preprocessing
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=5)

y = np.array(labels)

# Build LSTM Model
model = Sequential([
    Embedding(1000, 64, input_length=5),

    LSTM(64),

    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(X, y, epochs=10)

# Evaluate Model
loss, accuracy = model.evaluate(X, y)

print("Accuracy:", accuracy)

# Predict New Review
test = ["good quality"]

test_seq = tokenizer.texts_to_sequences(test)
test_seq = pad_sequences(test_seq, maxlen=5)

prediction = model.predict(test_seq, verbose=0)

print("Prediction Score:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Positive Sentiment")
else:
    print("Negative Sentiment")