import numpy as np
import matplotlib.pyplot as plt
import sys

def symmetrize(x):
    return x + x.T - np.diag(np.diag(x))

def harmadol(x):
    n = x.shape[0]
    m = int(n/3)
    print(m, n)
    y = np.zeros((m+1, m+1))
    for i in range(m+1):
        for j in range(m+1):
            y[i,j] = x[3*i, 3*j]
    return y

# ax = plt.imshow(symmetrize(ckamatrix), cmap='hot')
# plt.gca().invert_yaxis()
# plt.colorbar(ax)
# plt.savefig('heatmap.pdf')

ckamatrix = np.load('cka_matrix.npy')
y = symmetrize(ckamatrix)
#y = harmadol(y)

ax = plt.imshow(y, cmap='hot', vmin=0, vmax=1)
plt.gca().invert_yaxis()
plt.colorbar(ax)
plt.savefig('heatmap.pdf')


