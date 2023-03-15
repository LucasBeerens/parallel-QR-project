import numpy as np
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI

comm = MPI.COMM_WORLD

class BlockMatrix():
    def __init__(self, rowPartitions, columnPartitions):
        self.rowPartitions = rowPartitions
        self.columnPartitions = columnPartitions

        worldSize = comm.Get_size()

        if self.numberOfBlocks > worldSize:
            raise Exception("Number of blocks must not be bigger than world size.")

        rank = comm.Get_rank()

        self.blocks = np.zeros((len(rowPartitions), len(columnPartitions)))
        for i in range(self.numberOfBlocks):
            self.blocks[i % len(rowPartitions), i // len(columnPartitions)] = i

        self.index = (rank % len(rowPartitions), rank // len(columnPartitions))
        self.data = np.zeros((self.rowPartitions[self.index[0]], self.columnPartitions[self.index[1]]))
        comm.barrier()

    @property
    def numberOfBlocks(self):
        return len(self.rowPartitions) * len(self.columnPartitions)

    def fill(self):
        self.data[:,:] = np.random.rand(*self.data.shape)
        comm.barrier()


    def print(self):
        print(self.index)
        print(self.data)
        comm.barrier()

    def __add__(self, other):
        if self.rowPartitions != other.rowPartitions or self.columnPartitions != other.columnPartitions:
            raise Exception('Row and column partitions have to match for matrix addition')

        C = BlockMatrix(self.rowPartitions, self.columnPartitions)

        AData = comm.sendrecv(self.data, C.blocks[self.index])
        BData = comm.sendrecv(other.data, C.blocks[other.index])

        C.data = AData + BData

        comm.barrier()

        return C
    
    def __sub__(self, other):
        if self.rowPartitions != other.rowPartitions or self.columnPartitions != other.columnPartitions:
            raise Exception('Row and column partitions have to match for matrix addition')

        C = BlockMatrix(self.rowPartitions, self.columnPartitions)

        AData = comm.sendrecv(self.data, C.blocks[self.index])
        BData = comm.sendrecv(other.data, C.blocks[other.index])

        C.data = AData - BData

        comm.barrier()

        return C

    def __neg__(self):
        C = BlockMatrix(self.rowPartitions, self.columnPartitions)
        data = comm.sendrecv(self.data, C.blocks[self.index])
        C.data = -data

        comm.barrier()

        return C
    
    def __mul__(self, scalar):
        return self.multiplyWithScalar(scalar)
    
    def __rmul__(self, scalar):
        return self.multiplyWithScalar(scalar)
    
    def multiplyWithScalar(self, scalar):
        C = BlockMatrix(self.rowPartitions, self.columnPartitions)
        data = comm.sendrecv(self.data, C.blocks[self.index])
        C.data = scalar * data
        
        comm.barrier()

        return C
    
    def __matmul__(self, other):
        C = BlockMatrix(self.rowPartitions, other.columnPartitions)

        for j in range(len(C.columnPartitions)):
            comm.isend(self.data, C.blocks[self.index[0], j])

        for j in range(len(C.rowPartitions)):
            comm.isend(other.data, C.blocks[j, other.index[0]])

        for j in range(len(self.columnPartitions)):
            ABlock = comm.recv(source=self.blocks[C.index[0], j])
            print('received', C.index)
            BBlock = comm.recv(source=other.blocks[j, C.index[0]])
            C.data += ABlock @ BBlock

        comm.barrier()

        return C

    def full(self):
        allData = comm.gather([self.index, self.data])
        if comm.rank != 0:
            return

        fullMatrix = np.zeros((sum(self.rowPartitions), sum(self.columnPartitions)))

        for index, data in allData:
            rowFrom = sum(self.rowPartitions[:index[0]])
            rowTo = sum(self.rowPartitions[:index[0] + 1])
            columnFrom = sum(self.columnPartitions[:index[0]])
            columnTo = sum(self.columnPartitions[:index[0] + 1])
            fullMatrix[rowFrom:rowTo, columnFrom:columnTo] = data

        return fullMatrix