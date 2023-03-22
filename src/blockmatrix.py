import numpy as np
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI

globalIDCounter = [0]

def generateGlobalId():
    globalIDCounter[0] += 1
    return globalIDCounter[0]

class BlockMatrix():
    def __init__(self, rowPartitions, columnPartitions, comm=MPI.COMM_WORLD):
        self.id = generateGlobalId()
        self.rowPartitions = rowPartitions
        self.columnPartitions = columnPartitions
        self.comm = comm

        worldSize = self.comm.Get_size()

        if self.numberOfBlocks > worldSize:
            raise Exception("Number of blocks must not be bigger than world size.")

        rank = self.comm.Get_rank()

        self.blocks = np.zeros((len(rowPartitions), len(columnPartitions)))
        for i in range(self.numberOfBlocks):
            self.blocks[i % len(rowPartitions), i // len(rowPartitions)] = i

        if rank < self.numberOfBlocks:
            self.index = (rank % len(rowPartitions), rank // len(rowPartitions))
            self.data = np.zeros((self.rowPartitions[self.index[0]], self.columnPartitions[self.index[1]]))
        else:
            self.index = None
            self.data = None
            print('rank:{}'.format(rank))
            print('blocks:{}'.format(self.numberOfBlocks))
            print('\n')

        self.comm.barrier()

    @property
    def numberOfBlocks(self):
        return len(self.rowPartitions) * len(self.columnPartitions)

    def fill(self):
        if self.index is not None:
            self.data[:,:] = np.random.randint(low=0, high=10, size=self.data.shape)
        self.comm.barrier()


    def print(self):
        print(self.index)
        print(self.data)
        self.comm.barrier()

    def __add__(self, other):
        if self.rowPartitions != other.rowPartitions or self.columnPartitions != other.columnPartitions:
            raise Exception('Row and column partitions have to match for matrix addition')

        C = BlockMatrix(self.rowPartitions, self.columnPartitions)

        if self.index is not None:
            AData = self.comm.sendrecv(self.data, C.blocks[self.index])
        if other.index is not None:
            BData = self.comm.sendrecv(other.data, C.blocks[other.index])

        if C.index is not None:
            C.data = AData + BData

        self.comm.barrier()

        return C
    
    def __sub__(self, other):
        if self.rowPartitions != other.rowPartitions or self.columnPartitions != other.columnPartitions:
            raise Exception('Row and column partitions have to match for matrix addition')

        C = BlockMatrix(self.rowPartitions, self.columnPartitions)

        if self.index is not None:
            AData = self.comm.sendrecv(self.data, C.blocks[self.index])
        if other.index is not None:
            BData = self.comm.sendrecv(other.data, C.blocks[other.index])

        if C.index is not None:
            C.data = AData - BData

        self.comm.barrier()

        return C

    def __neg__(self):
        C = BlockMatrix(self.rowPartitions, self.columnPartitions)

        if self.index is not None:
            data = self.comm.sendrecv(self.data, C.blocks[self.index])

        if C.index is not None:
            C.data = -data

        self.comm.barrier()

        return C
    
    def __mul__(self, scalar):
        return self.multiplyWithScalar(scalar)
    
    def __rmul__(self, scalar):
        return self.multiplyWithScalar(scalar)
    
    def multiplyWithScalar(self, scalar):
        C = BlockMatrix(self.rowPartitions, self.columnPartitions)

        if self.index is not None:
            data = self.comm.sendrecv(self.data, C.blocks[self.index])

        if C.index is not None:
            C.data = scalar * data
        
        self.comm.barrier()

        return C
    
    def __matmul__(self, other):
        C = BlockMatrix(self.rowPartitions, other.columnPartitions)

        if self.index is not None:
            for j in range(len(C.columnPartitions)):
                self.comm.isend(self.data, C.blocks[self.index[0], j], tag=self.id)

        if other.index is not None:
            for j in range(len(C.rowPartitions)):
                self.comm.isend(other.data, C.blocks[j, other.index[1]], tag=other.id)

        if C.index is not None:
            for j in range(len(self.columnPartitions)):
                ABlock = self.comm.recv(source=self.blocks[C.index[0], j], tag=self.id)
                BBlock = self.comm.recv(source=other.blocks[j, C.index[1]], tag=other.id)
                C.data += ABlock @ BBlock

        self.comm.barrier()

        return C
    
    def subselect(self,rows,columns):
        rowPartitions = [self.rowPartitions[i] for i in rows]
        columnPartitions = [self.columnPartitions[i] for i in columns]
        C = BlockMatrix(rowPartitions, columnPartitions)

        for numi,i in enumerate(rows):
            for numj, j in enumerate(columns):
                if self.index == (i,j):
                    self.comm.isend(self.data, C.blocks[numi,numj], tag=numi*len(columns)+numj)
                if self.index == (numi,numj):
                    C.data = self.comm.recv(source=self.blocks[i,j],tag=self.index[0]*len(columns)+self.index[1])
        self.comm.barrier()
        return C

    def full(self):
        allData = self.comm.gather([self.index, self.data])
        if self.comm.rank != 0:
            return

        fullMatrix = np.zeros((sum(self.rowPartitions), sum(self.columnPartitions)))

        for index, data in allData:
            if index is None:
                continue

            rowFrom = sum(self.rowPartitions[:index[0]])
            rowTo = sum(self.rowPartitions[:index[0] + 1])
            columnFrom = sum(self.columnPartitions[:index[1]])
            columnTo = sum(self.columnPartitions[:index[1] + 1])
            fullMatrix[rowFrom:rowTo, columnFrom:columnTo] = data

        return fullMatrix