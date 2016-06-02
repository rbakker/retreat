from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import mixture
import matplotlib.pyplot as plt
import itertools
from scipy import linalg
import numpy as np

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# Run kmeans
km = KMeans(n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

# Run dbscan
db = DBSCAN(eps=0.2, min_samples=5, algorithm='auto', metric='euclidean')
y_db = db.fit_predict(X)

# Run EM
lowest_bic = np.infty
bic = []
n_components_range = range(1, 3)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a mixture of Gaussians with EM
        gmm = mixture.GMM(n_components=2, covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
clf = best_gmm
bars = []
y_gmm = clf.fit_predict(X)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.scatter(X[y_gmm == 0, 0], X[y_gmm == 0, 1], s=40, c='lightgreen', marker='o', label='cluster1')
ax1.scatter(X[y_gmm == 1, 0], X[y_gmm == 1, 1], s=40, c='lightblue', marker='s', label='cluster2')
ax1.set_title('EM')

ax2.scatter(X[y_db == 0, 0], X[y_db == 0, 1], s=40, c='lightgreen', marker='o', label='cluster1')
ax2.scatter(X[y_db == 1, 0], X[y_db == 1, 1], s=40, c='lightblue', marker='s', label='cluster2')
ax2.set_title('dbscan')


plt.legend()
plt.show()