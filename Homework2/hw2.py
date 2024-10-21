import numpy as np
from torchvision.datasets import MNIST


# Exercise 1 -------------------------------------------------
def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)

    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)

    return mnist_data, mnist_labels


train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

# Exercise 2 -------------------------------------------------
# Normalizing the data by ensuring that all values are in [0, 1]
train_X = np.array(train_X) / 255.0
test_X = np.array(test_X) / 255.0

# One-hot-encoding the labels
train_Y = np.array(train_Y)
train_Y_encoded = np.zeros((train_Y.size, train_Y.max() + 1))
train_Y_encoded[np.arange(train_Y.size), train_Y] = 1

test_Y = np.array(test_Y)
test_Y_encoded = np.zeros((test_Y.size, test_Y.max() + 1))
test_Y_encoded[np.arange(test_Y.size), test_Y] = 1


# Exercise 3 -------------------------------------------------
def create_batches(X, Y, batch_size=100):
    n = X.shape[0]
    for i in range(0, n, batch_size):
        yield X[i:i + batch_size], Y[i:i + batch_size]


def softmax(values):
    exp_values = np.exp(values - np.max(values, axis=1, keepdims=True))
    exp_values_sum = np.sum(exp_values, axis=1, keepdims=True)
    return exp_values / exp_values_sum


def cross_entropy(y_pred, y_true):
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss


def compute_gradients(X_batch, Y_batch, W, b, learning_rate):
    m = X_batch.shape[0]

    Z = np.dot(X_batch, W) + b
    probs = softmax(Z)

    loss = cross_entropy(probs, Y_batch)

    W = W + np.dot(X_batch.T, Y_batch - probs) / m
    b = b + learning_rate * np.sum(Y_batch - probs) / m

    return W, b, loss


def compute_accuracy(X, Y, W, b):
    Z = np.dot(X, W) + b
    probs = softmax(Z)
    predictions = np.argmax(probs, axis=1)
    correct_labels = np.argmax(Y, axis=1)

    accuracy = np.mean(predictions == correct_labels)
    return accuracy


W = np.random.random((784, 10)) * 0.01
b = np.zeros(10)

learning_rate = 0.1
epochs = 100

for epoch in range(epochs):
    for X_batch, Y_batch in create_batches(train_X, train_Y_encoded):
        W, b, batch_loss = compute_gradients(X_batch, Y_batch, W, b, learning_rate)

    if epoch % 10 == 0:
        train_accuracy = compute_accuracy(train_X, train_Y_encoded, W, b)
        test_accuracy = compute_accuracy(test_X, test_Y_encoded, W, b)
        print(
            f"Epoch {epoch}: Train Accuracy = {train_accuracy * 100:.2f}%, Test Accuracy = {test_accuracy * 100:.2f}%")

final_test_accuracy = compute_accuracy(test_X, test_Y_encoded, W, b)
print(f"Final Test Accuracy: {final_test_accuracy * 100:.2f}%")
