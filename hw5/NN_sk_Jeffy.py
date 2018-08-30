#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
INF 552 Homework 5
Feed Foward Neural Network using Backward Propagation
Group Members: Tianyu Zhang (zhan198),  Minyi Huang (minyihua),  Jeffy Merin Jacob (jeffyjac)
Date: 3/24/2018
Python 2.7
"""
import numpy as np
from sklearn.neural_network import MLPClassifier


def read_pgm(pgmf):
    assert pgmf.readline() == 'P5\n'
    pgmf.readline()
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255
    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    raster_arr = np.asarray(raster)
    return raster_arr


if __name__ == "__main__":

    # import train data
    train_result = []
    train_data = []
    with open('downgesture_train.list') as f:
        lines = f.read().splitlines()
    for line in lines:
        if 'down' in line:
            train_result.append(1.0)
        else:
            train_result.append(0.0)
    for i in range(184):
        f2 = open(lines[i], 'rb')
        train_data.append(read_pgm(f2))
    train_data = np.array(train_data, dtype="float").reshape([len(train_data), 32 * 30])
    train_data = train_data / 255

    # import test data
    test_data = []
    test_result = []
    with open('downgesture_test.list') as g:
        lines = g.read().splitlines()
    for line in lines:
        if 'down' in line:
            test_result.append(1.0)
        else:
            test_result.append(0.0)
    for i in range(83):
        g2 = open(lines[i], 'rb')
        test_data.append(read_pgm(g2))
    test_data = np.array(test_data, dtype="float").reshape([len(test_data), 32 * 30])
    test_data = test_data / 255
    print("")
    print("True Labels")
    print(np.array(test_result))
    print("")
    nn_clf = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='lbfgs', learning_rate='constant',
                           learning_rate_init=0.1, max_iter=1000)
    nn_clf.fit(train_data, train_result)
    print("Predicted Labels")
    print(nn_clf.predict(test_data))
    print("")
    print("Score")
    print(nn_clf.score(test_data, test_result))
