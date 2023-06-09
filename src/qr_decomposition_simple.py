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

    Y = np.zeros(A.shape)
    W = np.zeros(A.shape)
    
    for j in range(min(A.shape)):
        v, beta = householderReflection(R[j:,j:])
        
        if j == 0:
            Y[:,0] = v
            W[:,0] = -beta * v
        else:
            vHat = np.zeros(A.shape[0])
            vHat[j:] = v
            z = -beta * (vHat + W @ (Y.T @ vHat))
            Y[:,j] = vHat
            W[:,j] = z

        Qj = np.identity(A.shape[0])
        Qj[j:,j:] = np.identity(A.shape[0] - j) - beta * np.outer(v, v)

        # Can be done more efficiently by using the low-rank representation of Qj
        Q = Qj @ Q
        R = Qj @ R

    return Q.T, R, W, Y

if __name__ == 'main':

    A = np.random.rand(6, 8)
    Q, R, W, Y = qrDecomposition(A)

    QWithWY = np.identity(A.shape[0]) + W @ Y.T
    RWithWY = QWithWY.T @ A

    RWithWY *= (abs(RWithWY) > TOL)
    plt.spy(RWithWY)
    plt.show()

    print(np.linalg.norm(Q @ R - A))
    print(np.linalg.norm(Q - QWithWY))
    print(np.linalg.norm(R - RWithWY))
    print(np.linalg.norm(Q @ Q.T - np.identity(A.shape[0])))