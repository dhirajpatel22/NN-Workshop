# NN-Workshop
A simple introduction to neural networks workshop using TensorFlow and the MNIST handwritten digits dataset.

## Overview
This project implements a neural network-based digit classification system using TensorFlow and Keras. The model is trained on the MNIST dataset, which consists of handwritten digits (0–9), and learns to classify grayscale images based on pixel patterns. The goal of this project is to demonstrate fundamental deep learning concepts, including data preprocessing, one-hot encoding, neural network architecture design, training, evaluation, and prediction visualization.

## Requirements
Install all dependencies using:

```
pip install -r requirements.txt
```

## How It Works

1. Load Dataset: Loads the MNIST handwritten digit dataset from TensorFlow.
2. One-Hot Encode Labels: Converts digit labels into categorical vectors.
3. Normalize Image Data: Scales pixel values to the range [0, 1].
4. Build Neural Network: Creates a fully connected neural network.
5. Compile Model: Configures optimizer, loss function, and metrics.
6. Train Model: Trains the network on labeled training data.
7. Evaluate Model: Tests performance on unseen data.
8. Visualize Prediction: Displays a random test image and predicts its digit.

## Code Explanation
### 1. Import Libraries
```
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
```

* tensorflow: Provides tools for building and training neural networks.
* keras.layers & models: Used to define the neural network architecture.
* to_categorical: Converts numeric labels to one-hot encoded vectors.
* numpy: Handles numerical operations.
* matplotlib: Displays images and visual output.

### 2. Load Dataset
```   
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

* Loads the MNIST dataset directly from TensorFlow.
* Each image is 28 × 28 pixels.

### 3. One-Hot Encode Labels
```
y_train_oh = to_categorical(y_train, 10)
y_test_oh = to_categorical(y_test, 10)
```

* Converts digit labels (0–9) into vectors of length 10.
* Required because the model uses Mean Squared Error (MSE) loss.
* Example: Digit 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

### 4. Normalize Image Data
```
x_train = x_train / 255
x_test = x_test / 255
```

* Pixel values originally range from 0 to 255 (because they are grayscale).
* Change range to be from 0 to 1

### 5. Build Neural Network
```
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

* Flatten Layer:
  * Converts 28×28 images into a 1D vector.
* Hidden Layer:
  * 128 neurons
  * ReLU activation
* Output Layer:
  * 10 neurons (one per digit)
  * Softmax activation to produce probabilities.

### 6. Compile the Model
```
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)
```
* Optimizer: Adam (adaptive learning rate).
* Loss Function: Mean Squared Error (MSE).
* Metric: Accuracy.

### 7. Train the Model
```
model.fit(
    x_train,
    y_train_oh,
)
```
* Trains the model on train data and labels.

### 8. Evaluate the Model
```
test_loss, test_acc = model.evaluate(x_test, y_test_oh)
print("Test accuracy:", test_acc)
```
* Evaluates model performance on unseen test data.
* Outputs classification accuracy.

### 9. Visualize and Predict a Random Test Image
```
index = np.random.randint(0, len(x_test))
image = x_test[index]
true_label = y_test[index]
```
* Randomly selects an image from the test set.
```
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
```
* Displays the selected handwritten digit.

### 10. Make a Prediction
```
prediction = model.predict(image.reshape(1, 28, 28))
predicted_label = np.argmax(prediction)
```
* Model outputs probability scores for each digit (because of softmax).
* argmax selects the digit with the highest probability.

```
print("Predicted digit:", predicted_label)
print("Actual digit:", true_label)
```
* Displays predicted vs actual digit.

## Notes
* Mean Squared Error is not the standard loss for classification, but is used here for educational purposes.
* The model architecture is intentionally simple for clarity and learning.

## Conclusion

This project demonstrates a complete workflow for building and evaluating a neural network for image classification. It provides hands-on experience with TensorFlow, data preprocessing, neural network training, and prediction visualization using a well-known benchmark dataset.
