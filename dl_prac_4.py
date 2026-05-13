### To design and implement a CNN model for Image classification and optimize it using different hyperparameters
# Import Libraries
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load Dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Data Preprocessing
X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Different Hyperparameter Values
filters_list = [32, 64]
dropout_list = [0.5, 0.3]

# Training Different Models
for filters, dropout in zip(filters_list, dropout_list):

    print(f"\nTraining Model with Filters={filters} "
          f"and Dropout={dropout}")

    # CNN Model
    model = Sequential([
        tf.keras.Input(shape=(28,28,1)),

        Conv2D(filters, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(dropout),

        Dense(10, activation='softmax')
    ])

    # Compile Model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train Model
    model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=64,
        verbose=1
    )

    # Evaluate Model
    loss, accuracy = model.evaluate(X_test, y_test)

    print("Test Accuracy:", accuracy)