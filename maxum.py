import numpy as np
X = np.array([0, 1, 1, 0])


for i in range(X.size - 1, 0, -1):
    X[i - 1] = np.maximum(X[i - 1], X[i])

print(X)