import pylab as pl
import numpy as np
from sklearn import random_projection, manifold, decomposition, lda

def visualize(x, y, method):
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = (x-x_min)/(x_max-x_min)
    pl.figure()
    for c, i, target_name in zip("rg", [0, 1], ['negative', 'positive']):
        pl.scatter(x[y==i, 0], x[y==i, 1], c=c, label=target_name)
    pl.legend()
    pl.title('%s of DSLondon dataset' % method)

if __name__ == '__main__':
    train_x = np.loadtxt('data/train.csv', delimiter=',')
    train_y = np.loadtxt('data/trainLabels.csv', delimiter=',')
    n_neighbors = 5

    # Principal Component Analysis
    pca = decomposition.RandomizedPCA(n_components=2)
    pca_x = pca.fit_transform(train_x)
    visualize(pca_x, train_y, 'PCA')

    # Random Projection
    rp = random_projection.SparseRandomProjection(n_components=2, random_state=28)
    rp_x = rp.fit_transform(train_x)
    visualize(rp_x, train_y, 'Random Projection')

    # Linear Discriminant Analysis
    lda = lda.LDA(n_components=2)
    lda_x = lda.fit_transform(train_x, train_y)
    visualize(lda_x, train_y, 'LDA')

    # Isomap
    isomap = manifold.Isomap(n_neighbors, n_components=2)
    isomap_x = isomap.fit_transform(train_x)
    visualize(isomap_x, train_y, 'Isomap')

    # Locally Linear Embedding 
    lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard') 
    lle_x = lle.fit_transform(train_x)
    visualize(lle_x, train_y, 'LLE')

    # Multi-Dimensional Scalling
    mds = manifold.MDS(n_components=2, n_init=1, max_iter=100)
    mds_x = mds.fit_transform(train_x)
    visualize(mds_x, train_y, 'MDS')

    # Laplacian Eigenmap (A.K.A. Spectral Embedding)
    se = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
    se_x = se.fit_transform(train_x)
    visualize(se_x, train_y, 'Spectral Embedding')

    pl.show()
