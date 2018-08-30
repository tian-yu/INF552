import numpy as np
import cvxopt
from cvxopt import solvers
from cvxopt.solvers import qp

def loadData():
    data = np.loadtxt('linsep.txt', delimiter = ',')
    points = data[:,:2]
    labels = data[:,2]
    labels = np.reshape(np.array(labels),(100,1))
    return points,labels

def training(points, lables):
	l = points.shape[0]
	P = cvxopt.matrix(np.dot(lables, lables.T) * np.dot(points, points.T))
	Q = cvxopt.matrix(np.ones(l) * -1)
	G = cvxopt.matrix(np.diag(np.ones(l) * -1))
	H = cvxopt.matrix(np.zeros(l))
	A = np.reshape((lables.T),(1,l))
	A = cvxopt.matrix(A.astype(np.double))
	B = cvxopt.matrix(0.0)
	solution = qp(P, Q, G, H, A, B)
	alpha = solution['x']
	return alpha

def svm(points, alpha):
	sv = []
	svAlpha = []
	index = []
	for i in range(len(alpha)):
		if(alpha[i] > 0.01):
			svAlpha.append(alpha[i])
			index.append(i)
	for j in index:
		sv.append(points[j])
	return sv, svAlpha, index

def main():
	points, labels = loadData()
	alpha = training(points,labels)
	w = np.dot((np.vstack(alpha * labels)).T, points)
	sv,svAlpha,index = svm(points,alpha)
	b = 1./labels[index[0]] - np.dot((np.reshape(w.T,(1,2))), sv[0])
	print("weight = ", w, " b = ", b)
	print('The equation of the line of separation: %f x 1 + %f x 2 + %f = 0' %(w[0,0],w[0,1],b))

if __name__=="__main__":
	main()
