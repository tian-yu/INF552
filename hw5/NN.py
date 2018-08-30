"""
 INF 552 Homework 5
 Part 1
 Group Members: Tianyu Zhang (zhan198),  Minyi Huang (minyihua),  Jeffy Merin Jacob (jeffyjac)
 Date: 3/27/2018
 Programming Language: Python 3.6
"""


import numpy as np
import imageio

N_FEATURES = 30 * 32
N_HIDDEN_SIZE = 100
weight_random_low = -1
weight_random_high = 1
TRAINING_SIZE = 184
TESTING_SIZE = 83
EPOCHS = 1000
LR = 1 # learning rate

train_size = 184
test_size = 83
epochs = 1000


TRAINING_LIST_NAME = 'downgesture_train.list'
TESTING_LIST_NAME = 'downgesture_test.list'


def getXandY(filename, sample_size):
    with open(filename) as file:
        training_list = file.read().splitlines()
    # print(training_list)

    training_set_size = len(training_list)

    X = np.empty((0, N_FEATURES), float)

    for sample in training_list[:sample_size]:
        im = imageio.imread(sample) / 255.0
        X = np.vstack((X, im.flatten()))

    Y = np.zeros((training_set_size, 1))
    for i in range(training_set_size):
        if "down" in training_list[i]:
            Y[i] = 1
    Y = Y[:sample_size]
    return X, Y


class NNClassifier:
    def __init__(self, epochs=epochs, n_hidden_size=N_HIDDEN_SIZE, learning_rate=LR):
        self.epochs = epochs
        self.n_hidden_size = n_hidden_size
        self.n_features = N_FEATURES
        self.LR = learning_rate
        self.w1, self.w2 = self._init_weights()
        self.X_biased = self.S1 = self.act1 = self.S2 = self.act2= np.array([])
        self.grad1 = self.grad2 = self.delta1 = self.delta2 = np.array([])

    def _init_weights(self):
        w1 = np.random.uniform(weight_random_low, weight_random_high,
                               size=self.n_hidden_size * (self.n_features + 1))
        w1 = w1.reshape(self.n_features + 1, self.n_hidden_size)
        w2 = np.random.uniform(weight_random_low, weight_random_high,
                               size=1 * (self.n_hidden_size + 1))
        w2 = w2.reshape(self.n_hidden_size + 1, 1)
        return w1, w2

    def sigmod(self, s):
        return 1.0 / (1.0 + np.exp(-s))

    def sigmod_derivative(self, sig):
        # return 1.0 - sig ** 2
        return sig * (1.0 - sig)

    def add_bias(self, X):
        intercept = np.ones((X.shape[0], 1))
        X_new = np.hstack((intercept, X))
        return X_new

    def forward(self, X):
        self.X_biased = self.add_bias(X)
        self.S1 = np.dot(self.X_biased, self.w1)
        self.act1 = self.sigmod(self.S1)
        self.act1 = self.add_bias(self.act1)
        self.S2 = np.dot(self.act1, self.w2)
        self.act2 = self.sigmod(self.S2)

    def backprop(self, Y):
        self.delta2 = (self.act2 - Y) * self.sigmod_derivative(self.act2) * 2
        self.delta1 = self.sigmod_derivative(self.act1) * np.dot(self.delta2, self.w2.T)
        self.delta1 = self.delta1[:, 1:]
        grad2 = np.dot(self.act1.T, self.delta2) / self.delta2.shape[0]
        self.w2 -= self.LR * grad2
        grad1 = np.dot(self.X_biased.T, self.delta1) / self.delta1.shape[0]
        self.w1 -= self.LR * grad1

    def fit(self, X, Y):
        for i in range(self.epochs):
            self.forward(X)
            self.backprop(Y)
            if (i%100 == 0 ) and (i!= 0):
                error = np.mean((self.act2 - Y)**2)
                # print("error of {i} is {error}".format(i = i, error= error))
        return self

    def predict_probab(self, X):
        X_biased = self.add_bias(X)
        S1 = np.dot(X_biased, self.w1)
        act1 = self.sigmod(S1)
        act1 = self.add_bias(act1)
        S2 = np.dot(act1, self.w2)
        act2 = self.sigmod(S2)
        probability = act2
        return probability

    def predict_hard(self, X):
        probability = self.predict_probab(X)
        prediction = np.round(probability)
        return prediction

    def score(self, X, Y):
        prediction = self.predict_hard(X)
        print("\nTrue Labels\n{a}".format(a=Y.ravel()))
        print("\nPredict Labels\n{a}".format(a=prediction.ravel()))
        accuracy = (prediction == Y).sum().astype(float) / len(Y)
        return accuracy


X_train, Y_train = getXandY(TRAINING_LIST_NAME, train_size)
X_test, Y_test = getXandY(TESTING_LIST_NAME, test_size)
nn = NNClassifier()
nn.fit(X_train, Y_train)
# print("\nScore on training set:\n{a}".format(a=nn.score(X_train, Y_train)))
print("\nScore\n{a}".format(a=nn.score(X_test, Y_test)))

