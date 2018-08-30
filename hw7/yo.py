

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix,solvers
import pylab as pl

if __name__=="__main__":
    data=pd.read_csv("nonlinsep.txt",sep=',',header=None)
    dataarray=np.array(data)
    classifier = np.array(dataarray)[:, 2]
    classifier = np.resize(classifier, (100, 1))
    X = np.array(dataarray)[:,0:2]
    up_poly_array = np.zeros(shape=(100, 6))
    for i in range(100):
        up_poly_array[i][0] = 1
        up_poly_array[i][1] = X[i][0] **2
        up_poly_array[i][2] = X[i][1] ** 2
        up_poly_array[i][3] = np.sqrt(2) * (X[i][0])
        up_poly_array[i][4] = np.sqrt(2) * (X[i][1])
        up_poly_array[i][5] = np.sqrt(2) * (X[i][0] * X[i][0])

    P = matrix(np.dot(up_poly_array, up_poly_array.T) * np.dot(classifier, classifier.T))
    q = matrix(np.ones(100) * -1)
    G = matrix(np.diag(np.ones(100) * -1))
    h = matrix(np.zeros(100))
    b = matrix([0.0])
    A = matrix(classifier.T, (1, 100))
    sol = solvers.qp(P, q, G, h, A, b)
    alpha = sol['x']
    weight = 0.0
    for i in range(100):
        weight += alpha[i] * classifier[i] * up_poly_array[i]
    print ('weight=', weight)
    fcoord = []
    falpha = []
    fval = []
    for i in range(100):
        if (alpha[i] > 0.0001):
            falpha.append(alpha[i])
            fcoord.append(up_poly_array[i])
            fval.append(classifier[i])
    print ("alpha=", falpha)
    b = (1 / fval[0]) - np.dot(weight, fcoord[0])
    print ('b=',b)

    bias = b[0]

    # normalize
    norm = np.linalg.norm(weight)
    weight, bias = weight / norm, bias / norm


