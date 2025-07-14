
## Regression

[[Regression]] is a supervised learning problem where the goal is to learn a mapping from input variables (features) to continuous output values.

#### required imports

packages like `tensorflow`, `numpy` and `matplotlib` are required to be installed on the device the model is to be trained, these are used for training and analysing purpose.

[[Neural Networks and Tensor(flow)]]

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

### initialising a model

a sequential regression model can be setup using the `tf.keras.models.Sequential` class which accepts a list of arrays of layers as an argument. it can consist of the input, hidden and output layers.

```python
model = tf.keras.models.Sequential([
    # Input layer (expects one feature per sample)
    tf.keras.layers.Dense(1, input_shape=(1,), name="input_layer"),

    # Hidden layer with 100 units and ReLU activation
    tf.keras.layers.Dense(100, activation='relu', name="hidden_layer"),

    # Output layer with 1 unit (for predicting a continuous value)
    tf.keras.layers.Dense(1, name="output_layer")
])
```

### compiling the model

the `.compile` method on `Model` class is used to train and compile a model. it allows configurations like loss control, optimiser settings and metrics that should be tracked while training.

```python
model.compile(
    # This is the function the model tries to minimize during training.
    # MeanAbsoluteError (MAE) computes the average absolute difference between predicted and actual values.
    loss=tf.keras.losses.MeanAbsoluteError(),
    # Controls how the model updates its weights during backpropagation.
    # SGD (Stochastic Gradient Descent) updates weights incrementally using each batch.
    optimizer=tf.keras.optimizers.SGD(),
    # List of metrics to monitor during training and evaluation.
    # "mae" tracks the mean absolute error for performance monitoring.
    metrics=["mae"]
)
```

### fitting data to train on (and visualising)

the `.fit` method in the `Model` class can be used to fit data to train models. it accepts the data to train the model on and number of epochs to process before finalising it.

```python
X = tf.constant([1,2,3,4,5])
Y = tf.constant([5,10,15,20,25])

plt.scatter(X, Y) # graph
model.fit(X, Y, epochs=100)
```

### getting predictions

the `.predict` function can be used on the `Model` object instance to get predictions after fitting and training data on it. 

```python
model.fit(np.array([2]))
```

### improving the model

the model can be improved in various ways using different changes we can apply to the options and our data.
##### 1. Improve Model Architecture
- Add more hidden layers to increase learning capacity.
- Increase the number of neurons in each layer.
- Use activation functions like `relu`, `tanh`, or `leaky_relu` for better non-linear learning.
- Ensure the final layer has one unit with no activation for regression.
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
```
##### 2. Use a Better Optimizer
- Replace `SGD` with more advanced optimizers like `Adam`, `RMSprop`, or `Adagrad`.
- Tune the learning rate (e.g., 0.001 to 0.01).\
- A high learning rate may cause instability.
- A low learning rate may lead to slow convergence.
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
```
##### 3. Train for More Epochs
* Increase the number of training epochs to give the model more time to learn.
- Use early stopping to avoid overfitting by monitoring validation loss.
```python
model.fit(X, Y, epochs=500)
```
##### 4. Use More Data
- Train on more examples to improve generalization.
- Augment existing data if applicable (e.g., time series, image data)

### analysing the model

#### graphical analysis

graphs can be plotted for better visual analysis of the trained model features

```python
def analyse_model(X_train, Y_train, X_test, Y_test, predictions):
    plt.figure(figsize=(10,7))
    plt.scatter(X_train, Y_train, c="b", label="Training data")
    plt.scatter(X_test, Y_test, c="g", label="Testing data")
    plt.scatter(X_test, predictions, c="r", label="Predictions")
    plt.legend()
```

![[graphaaa.png]]

#### mean value analysis

```python
mae = tf.keras.metrics.MeanAbsoluteError()
mae.update_state(Y_test, predictions)
mae_result = mae.result().numpy()
mse = tf.keras.metrics.MeanSquaredError()
mse.update_state(Y_test, predictions)
mse_result = mse.result().numpy()
rmse_result = np.sqrt(mse_result)

print(f"MAE: {mae_result}, MSE: {mse_result}, RMSE: {rmse_result}")
```
