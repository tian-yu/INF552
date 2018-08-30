import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

input = np.loadtxt("linear-regression.txt", delimiter=',', usecols=(0, 1), unpack=True).T
print(input)

labels = np.loadtxt("linear-regression.txt", delimiter=',', usecols=(2), unpack=True).T
print(labels)



def showPrediction(input, labels, predication):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca()
    cm = plt.cm.get_cmap('bwr')
    error_map = np.array(predication == labels)
    print(error_map)
    error_map = error_map.astype(int)
    # print(error_map)
    ax.scatter(input[:, 0], input[:, 1],
               c=error_map, vmin=0, vmax=1, cmap=cm)
    plt.show()


sk_linear = LinearRegression()
sk_linear.fit(input, labels)
print("\nWeights of scikit-learn:\t")
print(np.append(sk_linear.intercept_, sk_linear.coef_))
print("\nAccuracy of scikit-learn:\t{a}".format(a = sk_linear.score(input, labels)))

# showPrediction(input, labels, sk_linear.predict(input))