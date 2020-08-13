import numpy as np


def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')


def preprocess(data):
    X = []
    Y = []

    for training_example in data:
        X.append(training_example[:-1].T)
        Y.append(training_example[-1])

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    Y = Y.reshape(Y.shape[0], 1)

    return X, Y


class LogicGatesNetwork:
    def __init__(self, x, y, input_dim=2, hidden_dim=2, output_dim=1, learning_rate=1):
        self.x = x
        self.y = y

        self.num_samples = self.x.shape[0]
        self.learning_rate = learning_rate

        assert x.shape == (self.num_samples, input_dim)

        self.w1 = np.random.randn(input_dim, hidden_dim)  # shape: input_dim * hidden_dim
        self.b1 = np.random.randn(1, hidden_dim)  # shape: 1 * hidden_dim

        self.w2 = np.random.randn(hidden_dim, output_dim)  # shape: hidden_dim * output_dim
        self.b2 = np.random.randn(1, output_dim)  # shape: 1 * output_dim

    def forward_prop(self, x):
        # x shape: num_samples_in_x * input_dim

        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.tanh(z1)  # a1 shape: num_samples_in_x * hidden_dim

        z2 = np.dot(a1, self.w2) + self.b2
        a2 = sigmoid(z2)  # a2 shape: num_samples_in_x * output_dim

        activations = {"a1": a1, "a2": a2}

        return activations

    def backward_prop(self, activations):
        a1 = activations["a1"]  # a1 shape: num_samples_in_x * hidden_dim
        a2 = activations["a2"]  # a2 shape: num_samples_in_x * output_dim

        # loss = 1/2 (A2^2 - Y^2)
        # derivative of loss = A2 - Y
        dz2 = a2 - self.y  # shape: num_samples_in_x * output_dim
        dw2 = (1/ self.num_samples) * np.dot(a1.T, dz2)  # shape: hidden_dim * output_dim
        db2 = (1 / self.num_samples) * np.sum(dz2, axis=0, keepdims=True)  # shape: 1 * output_dim

        dz1 = np.multiply(np.dot(dz2, self.w2.T), 1 - np.power(a1, 2))  # shape: num_samples_in_x * hidden_dim
        dw1 = (1 / self.num_samples) * np.dot(self.x.T, dz1)  # shape: input_dim * hidden_dim
        db1 = (1 / self.num_samples) * np.sum(dz1, axis=0, keepdims=True)  # shape: 1 * hidden_dim

        # update weights
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2

        return {
            "dw1": dw1,
            "db1": db1,
            "dw2": dw2,
            "db2": db2,
        }

    def train(self, num_iterations=1000):
        for i in range(num_iterations):
            activations = self.forward_prop(self.x)
            self.backward_prop(activations)

    def predict(self, x):
        activations = self.forward_prop(x)
        return activations["a2"]


if __name__ == '__main__':
    files = ["and.csv", "or.csv", "xor.csv"]
    print("Choose your network")
    choice = int(input("1. AND" + "\n" + "2. OR" + "\n" + "3. XOR" + "\n" + "Enter choice: "))
    myFile = load_data(files[choice-1])

    X, Y = preprocess(myFile)

    network = LogicGatesNetwork(X, Y)
    network.train()

    test_input = np.array([[1, 0]])
    print(f"TEST PREDICT {test_input} :: ", network.predict(test_input)[0][0])
