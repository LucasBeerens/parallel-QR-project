from blockmatrix import BlockMatrix
from mpi4py import MPI
import matplotlib.pyplot as plt

import numpy as np

from qr_decomposition_simple import qrDecomposition

MPI.Init()

# A = BlockMatrix([3, 2, 5], [2, 2])
A = BlockMatrix(5 * [40], 4 * [50])
# B = BlockMatrix([2, 2, 3], [2, 2])
A.fill()

fullMatrixA = A.full()
# B.fill()
# C = A + B

Q, R = A.qr()
fullMatrixR = R.full()
fullMatrixQ = Q.full()
# diag = R.diag()

if fullMatrixR is not None:
    fullMatrixR *= (abs(fullMatrixR) > 1e-6)
    plt.spy(fullMatrixR)
    plt.show()
    QSeq, RSeq, _, _ = qrDecomposition(fullMatrixA)

    print(np.max(abs(fullMatrixR - RSeq)))
    print(np.max(abs(fullMatrixA - fullMatrixQ @ fullMatrixR)))
    print(np.max(abs(np.identity(fullMatrixQ.shape[0]) - fullMatrixQ.T @ fullMatrixQ)))

# D = A @ B
# fullMatrixA = A.full()
# fullMatrixB = B.full()
# fullMatrixD = D.full()

#E = BlockMatrix([2,2],[2,2])
#E.fill()

#F = E.subselect([0,1],[0])

#fullE = E.full()
#fullF = F.full()

# if MPI.COMM_WORLD.rank == 0:
#     print(fullMatrixA)
#     print(fullMatrixB)
#     print(fullMatrixD)
#     print(fullMatrixA @ fullMatrixB)
#    print(fullE)
#    print(fullF)

MPI.Finalize()