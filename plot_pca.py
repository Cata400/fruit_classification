from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    color='red',
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


x = np.random.randn(1000)
y = x + 0.5 * np.random.randn(1000)

data = np.stack((x, y), axis=1)
print(data.shape)
pca = PCA(n_components=2)
data_transformed = pca.fit_transform(data)
print(pca.components_)

poly1 = np.poly1d(pca.components_[0])
poly2 = np.poly1d(pca.components_[1])

plt.scatter(x, y)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()