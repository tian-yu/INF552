#
# INF 552 Homework 3
# Part 1: PCA
# Group Members: Tianyu Zhang (zhan198),  Minyi Huang (minyihua),  Jeffy Merin Jacob (jeffyjac)
# Date: 2/27/2018
# Programming Language: Python 3.6
#

import numpy as np
from numpy import linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DIMENSION_IN = 3
DIMENSION_OUT = 2

def main():
    input = np.loadtxt("pca-data.txt")
    print("input=\n{input}\n".format(input=input))

    mean = input.mean(0)
    print("mean=\n{mean}\n".format(mean=mean))

    input_subtracted = input - mean
    print("input_subtracted=\n{input_subtracted}\n".format(input_subtracted=input_subtracted))
    t = input_subtracted.transpose()
    print("t=\n{t}\n".format(t=t))

    covariance = np.cov(t)
    print("covariance=\n{covariance}\n".format(covariance=covariance))
    eigenValues, eigenVectors = linalg.eig(covariance)

    print("Before sorting:\n ")
    print("eigenValues=\n{Values}\n".format(Values = eigenValues))
    print("eigenVectors=\n{Vectors}\n".format(Vectors = eigenVectors))

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    print("After sorting:\n ")
    print("eigenValues=\n{Values}\n".format(Values = eigenValues))
    print("eigenVectors=\n{Vectors}\n".format(Vectors = eigenVectors))

    pValues = eigenValues[range(DIMENSION_OUT)]
    pVectors = eigenVectors[:, range(DIMENSION_OUT)]
    print("pValues= \n{pValues}\n".format(pValues = pValues))
    print("pVectors= \n{pVectors}\n".format(pVectors = pVectors))

    output = np.matmul(input_subtracted, pVectors)
    print("output= \n{output}\n".format(output = output))

    # eigenVecrors are already normalised!
    print("Direction of the first two principal components=\n(each column is a vector)\n{pVectors}\n".format(pVectors=pVectors))
    # plotTwoComponents(input, mean, pValues, pVectors)
    x = output[:,0]
    y = output[:,1]
    fig, ax = plt.subplots()
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.scatter(x, y)
    # plt.scatter(x, y, color="red", s=5)
    plt.scatter(x, y, color="red", s=20)
    plt.title("PCA Result")
    plt.show()


def plotTwoComponents(input, mean, pValues, pVectors):
    fig = plt.figure()
    # ax = fig.gca(111, projection='3d')
    ax = fig.gca(projection='3d')
    for x,y,z in input[range(0, 6000, 10)]:
        ax.scatter(x, y, z, color="red", s=5)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    for length, vector in zip(pValues, pVectors.transpose()):
        v = vector * 5 * np.sqrt(length)
        ax.quiver(mean[0], mean[1], mean[2], v[0], v[1], v[2])

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-20, 20)
    plt.title("Directions of the First Two Principal Components")
    plt.show()


main()
