#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:44:31 2018

@author: jeffyjacob
"""

import pandas as pd
import numpy as np
import math


# import matplotlib.pyplot as plt
# from sklearn import mixture

# Initialize Random Ric's
def initialize_ric():
    np.random.seed(50)
    ric = np.random.randint(low=1, high=5, size=(N, K))
    for i in range(N):
        s = ric[i].sum()
        for j in range(K):
            ric_norm[i][j] = ric[i][j] / float(s)
    return ric_norm


# Calculate the amplitude
def calc_amp(ric_norm):
    t_ric = ric_norm.T
    for c in range(K):
        ric_sum = t_ric[c].sum()
        amp[c] = ric_sum / N
    return amp


# Calculate the mean
def calc_mean(ric_norm):
    t_ric = ric_norm.T
    coord1 = np.transpose(coord)
    mean = []
    for c in range(K):
        temp = 0.0
        ric_sum = t_ric[c].sum()
        for i in range(N):
            temp = temp + (ric_norm[i][c] * coord1[i])
        temp = temp / ric_sum
        mean.append(temp)
    mean = np.matrix(mean)
    return mean


# Calculate the covariance matrix
def calc_cov(ric_norm, mean):
    matrix1 = np.matrix(coord)
    for c in range(K):
        temp = 0.0
        ric_sum = sum(row[c] for row in ric_norm)
        for i in range(N):
            mean_matrix = matrix1[i] - mean[c]
            temp = temp + (ric_norm[i][c] * (mean_matrix.T.dot(mean_matrix)))
        temp = temp / ric_sum
        if (c == 1):
            covar1 = temp
        elif (c == 2):
            covar2 = temp
        else:
            covar3 = temp
    return covar1, covar2, covar3


# Calculate Multivariate Guassian distribution
def guassian(mean, covar):
    matrix1 = np.matrix(coord)
    det = np.linalg.det(covar)
    inv = np.linalg.inv(covar)
    pdf = []
    for i in range(N):
        x_minus_mean = matrix1[i] - mean
        ep = math.exp((-0.5) * x_minus_mean * inv * x_minus_mean.T)
        y = (((det) ** (-0.5)) * ep) / (2 * math.pi)
        pdf.append(y)
    return pdf


# Re-calculate Ric
def recalc_ric(amp, pdf, ric_norm):
    for i in range(N):
        deno = 0.0
        for c in range(K):
            deno = deno + amp[c] * pdf[c][i]
        for c in range(K):
            num = amp[c] * pdf[c][i]
            ric_norm[i][c] = num / deno
    return ric_norm


# Calculate log likelihood
def calc_convergance(amp, pdf):
    outer = 0.0
    for i in range(N):
        inner = 0.0
        for c in range(K):
            inner = inner + amp[c] * pdf[c][i]
        outer = outer + math.log(inner)
    # print outer
    return outer


# E-M Algorithm for GMM
def EM_GMM(ric_norm):
    pdf = []
    amplitude = calc_amp(ric_norm)
    mean = calc_mean(ric_norm)
    cov1, cov2, cov3 = calc_cov(ric_norm, mean)
    pdf.append(guassian(mean[0], cov1))
    pdf.append(guassian(mean[1], cov2))
    pdf.append(guassian(mean[2], cov3))
    log_likelihood = calc_convergance(amplitude, pdf)
    ric_recalc = recalc_ric(amplitude, pdf, ric_norm)

    print ("___________________")
    print ("log_likelihood:", log_likelihood)
    print ("amplitude:", amplitude)
    print ("mean:", mean)
    print ("cov1:", cov1)
    print ("cov2:", cov2)
    print ("cov3:", cov3)
    print (ric_norm.sum())
    print ("___________________")
    return log_likelihood, amplitude, mean, cov1, cov2, cov3, ric_recalc


if __name__ == "__main__":

    # Initializations
    # pd.options.display.float_format = '{:,.9f}'.format
    N = 150  # N datapoints
    K = 3  # K Clusters
    ric_norm = np.zeros((N, K))
    amp = [0.0, 0.0, 0.0]

    # read clusters.txt
    coord = pd.read_csv('clusters.txt', sep=",", header=None)
    # Initialize ric and normalize

    ric_norm = initialize_ric()

    # prev_likelihood,prev_amp,prev_mean,prev_cov1,prev_cov2,prev_cov3,prev_ric_recalc stored inside prev_result
    prev_result = EM_GMM(ric_norm)
    prev_likelihood = prev_result[0]
    # curr_likelihood,curr_amp,curr_mean,curr_cov1,curr_cov2,curr_cov3,curr_ric_recalc stored inside curr_result
    curr_result = EM_GMM(prev_result[6])
    curr_likelihood = curr_result[0]
    ric_recalc = curr_result[6]

    while (abs(prev_likelihood) - abs(curr_likelihood) > 0.0001):
        prev_likelihood = curr_likelihood
        curr_result = EM_GMM(ric_recalc)
        curr_likelihood = curr_result[0]
        ric_recalc = curr_result[6]

        # Print Mean, Amplitude, Covariance
    if (abs(prev_likelihood) - abs(curr_likelihood) < 0.0001):
        print ("Guassian 1")
        print ("----------")
        print ("Mean      :", curr_result[2][0])
        print ("Amplitude :", curr_result[1][0])
        print ("Covariance:")
        print (curr_result[3])
        print ("Guassian 2")
        print ("----------")
        print ("Mean      :", curr_result[2][1])
        print ("Amplitude :", curr_result[1][1])
        print ("Covariance:")
        print (curr_result[4])
        print ("Guassian 3")
        print ("----------")
        print ("Mean      :", curr_result[2][2])
        print ("Amplitude :", curr_result[1][2])
        print ("Covariance:")
        print (curr_result[5])

"""      
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(coord)  
x = np.linspace(-4., 8.)
y = np.linspace(-4., 8.)
X, Y = np.meshgrid(x, y)
Z = (np.array(pdf)).ravel()
Z = Z.reshape(X.shape)
plt.contour(Z)
plt.scatter(coord[0], coord[1])
plt.show()
"""
