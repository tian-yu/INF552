#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
INF 552 Homework 4
Pocket Algorithm
Group Members: Tianyu Zhang (zhan198),  Minyi Huang (minyihua),  Jeffy Merin Jacob (jeffyjac)
Date: 3/09/2018
Python 2.7
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def predict(w, datapoint):
    x = np.matrix(datapoint)
    y = w * x.T
    if y > 0:
        return 1
    else:
        return -1


if __name__ == "__main__":
    N = 2000
    np.random.seed(25)
    first_col = []
    w = []
    count = 0
    c = 0
    misclass = []
    ypredict = np.zeros(2000)
    weight_matrix = np.zeros((2000, 4))
    # read into DataFrame
    dataset = pd.read_csv("classification.txt", sep=",", header=None)
    for i in range(N):
        first_col.append(1)
    df = {'x0': first_col, 'x1': dataset[0], 'x2': dataset[1], 'x3': dataset[2]}
    x = pd.DataFrame(data=df)
    datapoints_x = x.as_matrix()
    yactual = dataset[4].as_matrix()

    # intial random values of weight
    for i in range(4):
        w.append(np.random.uniform(0.01, 0.1))
    weights = np.asarray(w)

    # Training step
    for e in range(7000):
        count = 0
        weight_matrix = np.zeros((2000, 4))
        for i in range(2000):
            ypredict[i] = predict(weights, datapoints_x[i])
            if (ypredict[i] != yactual[i]):
                if (ypredict[i] < yactual[i]):
                    weights = weights + (0.01) * datapoints_x[i]
                    count = count + 1
                else:
                    weights = weights - (0.01) * datapoints_x[i]
                    count = count + 1
        print("Misclassifications at ", e, " : ", count)
        misclass.append(count)

    # Print weights,misclassifications and plot graph
    plt.scatter(list(range(7000)), misclass, s=5)
    plt.show()
    plt.title("Pocket Algorithm")
    print("Weights:", weights)
    for j in range(2000):
        if (ypredict[j] != yactual[j]):
            c = c + 1;
    print("Accuracy:")
    print("The lowest number of miss-classifications (", min(s for s in misclass), ") is at iteration ",
          misclass.index(min(s for s in misclass)))
