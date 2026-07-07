import numpy as np
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

Y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(42)
input_size = 2
hidden1 = 4
hidden2 = 4
output_size = 1
W1 = np.random.randn(input_size, hidden1)
b1 = np.zeros((1, hidden1))

W2 = np.random.randn(hidden1, hidden2)
b2 = np.zeros((1, hidden2))

W3 = np.random.randn(hidden2, output_size)
b3 = np.zeros((1, output_size))
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
lr = 0.01
epochs = 10000

for epoch in range(epochs):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    A3 = sigmoid(Z3)
    loss = np.mean((Y - A3) ** 2)
    dA3 = (A3 - Y)
    dZ3 = dA3 * A3 * (1 - A3)

    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * relu_derivative(Z2)

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    W3 -= lr * dW3
    b3 -= lr * db3

    W2 -= lr * dW2
    b2 -= lr * db2

    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss = {loss:.4f}")
pred = sigmoid(relu(relu(X @ W1 + b1) @ W2 + b2) @ W3 + b3)
print("\nPredictions:")
print(pred)

print("\nBinary Output:")
print((pred > 0.5).astype(int))
