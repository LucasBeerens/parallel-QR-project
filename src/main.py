from blockmatrix import BlockMatrix
from mpi4py import MPI

from qr_decomposition_simple import qrDecomposition

MPI.Init()

A = BlockMatrix([3, 2, 5], [4])
# B = BlockMatrix([2, 2, 3], [2, 2])
A.fill()
fullMatrixA = A.full()

# B.fill()
# C = A + B

A.qr()

if fullMatrixA is not None:
    qrDecomposition(fullMatrixA)
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