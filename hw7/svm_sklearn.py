
# coding: utf-8

# In[19]:


import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import normalize


def readFile(input_file_name):
    x = np.loadtxt(input_file_name, delimiter=',', usecols=(0, 1), unpack=True)
    y = np.loadtxt(input_file_name, delimiter=',', usecols=(2), unpack=True)
    return x, y

# input_file_name = "linsep.txt"
# x, y = readFile(input_file_name)
# clf = SVC(kernel='linear')
# clf.fit(x.T, y)
# print(clf.support_ )

input_file_name = "nonlinsep.txt"
x, y = readFile(input_file_name)
x = normalize(x)
print(x)
clf = SVC(kernel='rbf')
clf.fit(x.T, y)
print(clf.support_)
print(clf.dual_coef_)

