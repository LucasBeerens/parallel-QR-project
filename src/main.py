from blockmatrix import BlockMatrix
from mpi4py import MPI

MPI.Init()

A = BlockMatrix([2, 2], [2, 2])
B = BlockMatrix([2, 2], [2, 2])
A.fill()
B.fill()
C = A + B

tmp = C - A - B

fullMatrix = tmp.full()

if fullMatrix is not None:
    print(fullMatrix)

MPI.Finalize()