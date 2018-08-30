"""
 INF 552 Homework 5
 Part 2
 Group Members: Tianyu Zhang (zhan198),  Minyi Huang (minyihua),  Jeffy Merin Jacob (jeffyjac)
 Date: 3/27/2018
 Programming Language: Python 3.6
"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import imageio


N_FEATURES = 30 * 32
N_HIDDEN_SIZE = 100
weight_random_low = -0.01
weight_random_high = 0.01
TRAINING_SIZE = 184
TESTING_SIZE = 83
EPOCHS = 1000
LR = 0.1 # learning rate

TRAINING_LIST_NAME = 'downgesture_train.list'
TESTING_LIST_NAME = 'downgesture_test.list'

def getXandY(filename, sample_size):
    with open(filename) as file:
        training_list = file.read().splitlines()
    training_set_size = len(training_list)

    X = np.empty((0, N_FEATURES), float)

    for sample in training_list[:sample_size]:
        im = imageio.imread(sample) / 255.0
        # print(im)
        X = np.vstack((X, im.flatten()))

    Y = np.zeros((training_set_size, 1))
    for i in range(training_set_size):
        if "down" in training_list[i]:
            Y[i] = 1
    Y = Y[:sample_size]
    return X, Y


X_train, Y_train = getXandY(TRAINING_LIST_NAME, TRAINING_SIZE)
X_test, Y_test = getXandY(TESTING_LIST_NAME, TESTING_SIZE)

mlp = MLPClassifier(solver='lbfgs', alpha=1e-15,
                     hidden_layer_sizes=(N_HIDDEN_SIZE,), activation='logistic', learning_rate_init=LR, max_iter = EPOCHS)

mlp.fit(X_train, Y_train.ravel())
# print("Accuracy for train:\n{a}".format(a=mlp.score(X_train, Y_train.ravel())))
# print("Accuracy for test:\n{a}".format(a=mlp.score(X_test, Y_test.ravel())))
print("\nTrue Labels:\n{a}".format(a=Y_test.ravel()))
print("\nPredict Labels:\n{a}".format(a=mlp.predict(X_test)))
print("\nScore Labels:\n{a}".format(a=mlp.score(X_test, Y_test.ravel())))