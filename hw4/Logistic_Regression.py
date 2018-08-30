#
# INF 552 Homework 4
# Part: Logistic Regression
# Group Members: Tianyu Zhang (zhan198),  Minyi Huang (minyihua),  Jeffy Merin Jacob (jeffyjac)
# Date: 3/09/2018
# Programming Language: Python 3.6
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression

number_of_iterations = 7000
input = np.loadtxt("classification.txt", delimiter=',', usecols=(0, 1, 2), unpack=True).T
print(input)

labels = np.transpose(np.loadtxt("classification.txt", delimiter=',', usecols=(4), unpack=True))
print(labels)

def showData(input, labels):
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca(projection='3d')
    cm = plt.cm.get_cmap('PiYGh4')
    ax.scatter(input[:, 0], input[:, 1], input[:, 2],
                c = labels, vmin=-1.2, vmax=1.1, cmap=cm)
    plt.show()

def sigmoidFunction(s):
    return np.exp(s) / (1 + np.exp(s))

def E_out(input, labels, weights):
    dot_product = np.dot(input, weights)
    result = (input.shape[1]) * np.sum(np.log(1 + np.exp((-1) * labels * dot_product)))
    return result

# def getGradient(input, labels, weights):
#     dot_product = np.dot(input, weights)
#     item_1 = 1 / (1 + np.exp((-1) * labels * dot_product))
#     item_2 = (-1) * np.exp((-1) * labels * dot_product)
#     item_3 = labels[:, np.newaxis] * input
#     # item_3 = np.dot(labels, input)
#     gradient = (input.shape[1]) * np.sum((item_1 * item_2)[:, np.newaxis] * item_3)
#     return gradient

def getGradient(input, labels, weights):
    dot_product = np.dot(input, weights)
    item_1 = 1 / (1 + np.exp((-1) * np.dot(labels, dot_product)))
    item_2 = (-1) * np.exp((-1) * np.dot(labels, dot_product))
    # item_3 = labels[:, np.newaxis] * input
    item_3 = np.dot(labels, input)
    gradient = (input.shape[1]) * np.sum(np.dot(np.dot(item_1, item_2), item_3))
    return gradient

def logisitic_regresssion_loop(input, labels, number_of_iterations, learning_rate, with_intercept=False):
    if with_intercept :
        intercept = np.ones((input.shape[0], 1))
        input = np.hstack((intercept, input))

    weights = np.random.rand(input.shape[1])
    for i in range(number_of_iterations):
        weights -= learning_rate * getGradient(input, labels, weights)

        # Printing E_out value for debugging
        # if i % 1000 == 0:
        #     print (E_out(input, labels, weights))
    return input, weights


# def calculateAccuracy(input, labels, weights):
#     predication = np.round(sigmoidFunction(np.dot(input, weights)))
#     accuracy = (predication == labels).sum().astype(float) / len(predication)
#     return accuracy

# def calculateAccuracy(input, labels, weights):
#     dot_product = np.dot(input, weights)
#     s = labels[:, np.newaxis] * dot_product
#     accuracy = np.exp(np.sum(np.log(sigmoidFunction(s))))
#     return accuracy

def calculateAccuracy(input, labels, weights):
    dot_product = np.dot(input, weights)
    s = labels[:, np.newaxis] * dot_product
    accuracy = np.mean(sigmoidFunction(s))
    return accuracy

def showPrediction(input, labels, predication):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection='3d')
    cm = plt.cm.get_cmap('bwr')
    error_map = np.array(predication != labels)
    error_map = error_map.astype(int)
    ax.scatter(input[:, 0], input[:, 1], input[:, 2],
               c=error_map, vmin=0, vmax=1, cmap=cm)
    plt.show()

input_intercepted, weights = logisitic_regresssion_loop(input, labels, number_of_iterations, learning_rate=0.00001, with_intercept=True)
accuracy = calculateAccuracy(input_intercepted, labels, weights)

sk_logistic = LogisticRegression(fit_intercept=True, C = 1e16)
sk_logistic.fit(input, labels)


print("\nAfter {number_of_iterations} iterations\n".format(number_of_iterations = number_of_iterations))
print("Weights of our model:\t")
print(weights)
print("Weights of scikit-learn:\t")
print(np.append(sk_logistic.intercept_, sk_logistic.coef_[0]))
print("\n")

print("Accuracy of our model:\t{a}".format(a = accuracy))
# print(clf.n_iter_)
print("Accuracy of scikit-learn:\t{a}".format(a = sk_logistic.score(input, labels)))

# showPrediction(input, labels, np.round(sigmoidFunction(np.dot(input_intercepted, weights))))
# showPrediction(input, labels, sk_logistic.predict(input))