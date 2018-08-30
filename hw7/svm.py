"""
INF 552 Homework 7
Support Vectors Machine
Part2
Group Members: Tianyu Zhang (zhan198),  Minyi Huang (minyihua),  Jeffy Merin Jacob (jeffyjac)
Date: 04/28/2018
Python 3.6
"""

import cvxopt as cvxopt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def readFile(input_file_name):
    x = np.loadtxt(input_file_name, delimiter=',', usecols=(0, 1), unpack=True)
    y = np.loadtxt(input_file_name, delimiter=',', usecols=(2), unpack=True)
    return x, y


def plotGraph(x, y, svm):
    def linear_line(x, w, b):
        return (-w[0] * x - b) / w[1]
#     def poly_curve()
    
    positive_indices = np.where(y > 0)[0]
    negative_indices = np.where(y < 0)[0]
    
    plt.scatter(x[0, positive_indices], x[1, positive_indices], c='red', alpha=0.8, marker='o')
    plt.scatter(x[0, negative_indices], x[1, negative_indices], c='blue', alpha=0.8, marker='o')
    plt.scatter(x[0, svm.indices], x[1, svm.indices], c='yellow', alpha=0.5, s=300, marker='*')
    
    if svm.kernel_mode == "linear":
        start_x1 = - 0.4
        start_x2 = linear_line(start_x1, svm.w, svm.b)
        end_x1 = 1.0
        end_x2 = linear_line(end_x1, svm.w, svm.b)
        plt.plot([start_x1, end_x1], [start_x2, end_x2], "k")
    print("\nA graph is on show. Close the graph window to continue.\n")
    plt.show()
    
    

class SVM(object):
    threshold_linear = 1e-5
    threshold_poly = 1e-5
    threshold_rbf = 1e-5
    def __init__(self, kernel_mode = "linear"):
        self.kernel_mode = kernel_mode
        
    def fit(self, x, y):
        N = x.shape[1]
        P = self.createQ(x, y)
        q = np.ones(N)* -1.0
        G = np.diag(np.ones(N) * -1.0)
        h = np.zeros(N)
        A = y.reshape((1, N))
        b = 0.0
        self.a = self.QP_solver(P, q, G, h, A, b)
        self.getSVs()
        self.solveLine(x, y)
        self.printReport(x)
        
    def createQ(self, x, y):
        if self.kernel_mode == "linear":
            k = self.linear(x, x)
        elif self.kernel_mode == "rbf":
            k = self.rbf(x, x)
        elif self.kernel_mode == "poly":
            k = self.poly(x, x)
        return np.outer(y, y) * k

    def linear(self, x1, x2):
        N = x1.shape[1]
        k = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                k[i, j] = np.dot(x1[:, i].T, x2[:, j])
        return k
    #     return np.dot(x1.T, x2)


    def rbf(self, x1, x2):
        n_features = x1.shape[0]
        N = x1.shape[1]
        k = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                k[i, j] = np.exp(- np.power(np.linalg.norm(x1[:, i] - x2[:, j]), 2) / n_features)
        return k


    def poly(self, x1, x2):
        N = x1.shape[1]
        k = np.zeros((N, N))
        for i in range(N):
            z1 = np.array([1, x1[0, i]**2, x1[1, i]**2, np.sqrt(2)*x1[0, i], np.sqrt(2)*x1[1, i], np.sqrt(2)*x1[0, i]*x1[1, i]])
            for j in range(N):
                z2 = np.array([1, x2[0, j]**2, x2[1, j]**2, np.sqrt(2)*x2[0, j], np.sqrt(2)*x2[1, j], np.sqrt(2)*x2[0, j]*x2[1, j]])
                k[i, j] = np.dot(z1, z2)
        return k


    def solveLine(self, x, y):
        if self.kernel_mode == "linear":
            self.w = np.sum(self.a_filtered * y * x,  axis=1)
            self.b = np.mean(y[self.indices] - np.dot(self.w, x[:, self.indices]))
        elif self.kernel_mode == "rbf":
            self.w = None
            self.b = None
            return
        elif self.kernel_mode == "poly":
            z_sv = np.array([np.ones(self.a.shape[0]), x[0, :]**2, x[1, :]**2, np.sqrt(2)*x[0, :], np.sqrt(2)*x[1, :], np.sqrt(2)*x[0, :]*x[1, :]])
            self.w = np.sum(self.a_filtered * y * z_sv,  axis=1)
            self.b = np.mean(y[self.indices] - np.dot(self.w, z_sv[:, self.indices]))
            print(self.b)


    def getSVs(self):
        if self.kernel_mode == "linear":
            thre = SVM.threshold_linear
        elif self.kernel_mode == "rbf":
            thre = SVM.threshold_rbf
        elif self.kernel_mode == "poly":
            thre = SVM.threshold_poly
        self.a_filtered = np.array(self.a)
        self.a_filtered[self.a_filtered < thre] = 0
        self.indices = np.where(self.a_filtered > 0)[0]
        
    
    def printReport(self, x):
        print("\nChosen mode: {mode} kernel mode".format(mode = self.kernel_mode))
        print("\nSupport Vectors Indices:\n{sv}".format(sv = self.indices))
        print("\nSupport Vectors:\n{sv}".format(sv = x[:, self.indices].T))
        if self.w is not None:
            print("\nweights = {w}\nbias = {b}".format(w = self.w, b = self.b))
        else:
            print("\nrdf kernel mode doesn't support weights and bias")

    def QP_solver(self, P, q, G=None, h=None, A=None, b=None):
	    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
	    if G is not None:
	        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
	        if A is not None:
	            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
	    sol = cvxopt.solvers.qp(*args)
	    if 'optimal' not in sol['status']:
	        return None
	    return np.array(sol['x']).reshape((P.shape[1],))
            


def main():
    print("INF Homework 7 SVM started successfully\n")
    quit = False
    while(not quit):
        response = input("Please enter the kernel function mode ('linear' / 'poly' / 'rbf') or enter 'quit' to quit: ")
        if "quit" in response:
            print("\nThank you for using. Have a good day")
            quit = True
        elif "linear" in response:
            input_file_name = "linsep.txt"
            x, y = readFile(input_file_name)
            linear_svm = SVM("linear")
            linear_svm.fit(x, y)
            plotGraph(x, y, linear_svm)
        elif response in ["poly" , "rbf"] :
            input_file_name = "nonlinsep.txt"
            x, y = readFile(input_file_name)
            if response in "rbf": 
            	x = normalize(x)
            linear_svm = SVM(response)
            linear_svm.fit(x, y)
            plotGraph(x, y, linear_svm)
        else:
            print("Invalid input\n")
            continue
main()

