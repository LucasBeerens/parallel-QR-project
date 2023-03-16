import numpy as np
import matplotlib.pyplot as plt

TOL = 1e-8

def householderReflection(A):
    x = A[:,0]
    n = len(x)

    sigma = np.dot(x[1:n], x[1:n])

    v = np.zeros(n)
    v[0] = 1
    v[1:n] = x[1:n]

    if abs(sigma) < TOL:
        beta = 0
    else:
        mu = np.sqrt(x[0]**2 + sigma)
        if x[0] <= 0:
            v[0] = x[0] - mu
        else:
            v[0] = -sigma / (x[0] + mu)
        
        beta = 2 * v[0]**2 / (sigma + v[0]**2)
        v = v / v[0]

    return v, beta

def qrDecomposition(A):
    Q = np.identity(A.shape[0])
    R = A.copy()
    
    for j in range(A.shape[1]):
        v, beta = householderReflection(R[j:,j:])
        Qj = np.identity(A.shape[0])

        Qj[j:,j:] = np.identity(A.shape[0] - j) - beta * np.outer(v, v)

        Q = Qj @ Q
        R = Qj @ R

    return Q.T, R

A = np.random.rand(10, 10)
Q, R = qrDecomposition(A)

R *= (abs(R) > TOL)
plt.spy(R)
plt.show()

print(np.linalg.norm(Q @ R - A))