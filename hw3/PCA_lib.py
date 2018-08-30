import numpy as np
from sklearn.decomposition import PCA

input = np.loadtxt("pca-data.txt")
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(input)


print(pca.components_)

print(pca.explained_variance_)