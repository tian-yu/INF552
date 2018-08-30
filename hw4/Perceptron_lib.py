import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Perceptron

input = np.loadtxt("classification.txt", delimiter=',', usecols=(0, 1, 2), unpack=True).T
print(input)

labels = np.loadtxt("classification.txt", delimiter=',', usecols=(4), unpack=True).T
print(labels)


def showPrediction(input, labels, predication):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection='3d')
    cm = plt.cm.get_cmap('bwr')
    error_map = np.array(predication != labels)
    print(error_map)
    error_map = error_map.astype(int)
    # print(error_map)
    ax.scatter(input[:, 0], input[:, 1], input[:, 2],
               c=error_map, vmin=0, vmax=1, cmap=cm)
    plt.show()


sk_perceptron = Perceptron(max_iter=7000)
sk_perceptron.fit(input, labels)
print("\nWeights of scikit-learn:\t")
print(np.append(sk_perceptron.intercept_, sk_perceptron.coef_))
print("\nAccuracy of scikit-learn:\t{a}".format(a = sk_perceptron.score(input, labels)))
print("\niter:\t{a}".format(a = sk_perceptron.n_iter_))


showPrediction(input, labels, sk_perceptron.predict(input))