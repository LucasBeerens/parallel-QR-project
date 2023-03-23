import numpy as np
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI

TOL = 1e-8

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
            
        self.comm.barrier()

    @property
    def numberOfBlocks(self):
        return len(self.rowPartitions) * len(self.columnPartitions)

    def qr(self):
        R = self.copy()

        for j in range(len(self.columnPartitions)):
            isFocusedColumn = self.index is not None and self.index[1] == j
            subCommunicator = self.comm.Split(0 if isFocusedColumn else MPI.UNDEFINED, self.index[0] if self.index is not None else MPI.UNDEFINED)
            
            W, Y = None, None

            if isFocusedColumn:
                W, Y = self.qrAux(R, j, subCommunicator)
            
            self.comm.barrier()

            self.applyWY(R, W, Y, j)

        self.comm.barrier()

        return R

    def applyWY(self, R, W, Y, columnPartitionIndex):
        if W is not None and Y is not None:
            for j in range(columnPartitionIndex + 1, len(self.columnPartitions)):
                self.comm.isend((W, Y), self.blocks[self.index[0], j])

        row = MPI.UNDEFINED if self.index is None else self.index[0]
        column = MPI.UNDEFINED if self.index is None else self.index[1]
        columnComm = self.comm.Split(column, row)

        if self.index is not None and self.index[1] > columnPartitionIndex:
            W, Y = self.comm.recv(source=self.blocks[self.index[0], columnPartitionIndex])
            print(self.index)
            print(W)
            WRLocal = W.T @ R.data
            WR = columnComm.allreduce(WRLocal)
            R.data += Y @ WR

        self.comm.barrier()
            

    def qrAux(self, R, columnPartitionIndex, localComm):
        Y = np.zeros(self.data.shape)
        W = np.zeros(self.data.shape)
        
        columnOffset = sum(self.columnPartitions[:columnPartitionIndex])

        for j in range(columnOffset, min(columnOffset + self.columnPartitions[columnPartitionIndex], sum(self.rowPartitions))):
            v, beta = self.householderReflectionAux(R, localComm, self.index[1], j)
            
            if j - columnOffset == 0:
                Y[:,0] = v
                W[:,0] = -beta * v
            else:
                localYv = Y.T @ v
                Yv = localComm.allreduce(localYv)

                z = -beta * (v + W @ Yv)
                Y[:,j - columnOffset] = v
                W[:,j - columnOffset] = z

            vR = v @ R.data
            K = localComm.allreduce(vR)
            R.data -= beta * np.outer(v, K)

        return W, Y

    def householderReflectionAux(self, R, localComm, columnPartition, j):
        rowOffset = sum(self.rowPartitions[:self.index[0]])
        rowEnd = sum(self.rowPartitions[:self.index[0] + 1])

        columnOffset = sum(self.columnPartitions[:columnPartition])
        relativeColumn = j - columnOffset
        
        focusedBlockRank = None
        totalRowOffset = 0
        for i in range(len(self.rowPartitions)):
            if j >= totalRowOffset and j < totalRowOffset + self.rowPartitions[i]:
                focusedBlockRank = i
                break

            totalRowOffset += self.rowPartitions[i]
        
        focusedBlockRank = len(self.rowPartitions) - 1 if focusedBlockRank is None else focusedBlockRank

        isFocusedBlock = j >= rowOffset and j < rowEnd
        localFocusedIndex = j - rowOffset if isFocusedBlock else None

        x = np.zeros(self.data.shape[0])
        if j < rowEnd:
            k = max(0, (j - rowOffset))
            x[k:] = R.data[k:,relativeColumn]

        localSigma = np.dot(x, x)

        if isFocusedBlock:
            localSigma -= x[localFocusedIndex]**2

        sigma = localComm.allreduce(localSigma)

        v = x.copy()

        if isFocusedBlock:
            v[localFocusedIndex] = 1
        
        if abs(sigma) < TOL:
            beta = 0
        else:
            beta = None

            if isFocusedBlock:
                mu = np.sqrt(x[localFocusedIndex]**2 + sigma)
                if x[localFocusedIndex] <= 0:
                    v[localFocusedIndex] = x[localFocusedIndex] - mu
                else:
                    v[localFocusedIndex] = -sigma / (x[localFocusedIndex] + mu)
                
                beta = 2 * v[localFocusedIndex]**2 / (sigma + v[localFocusedIndex]**2)
            
            scalingFactor = localComm.bcast(v[localFocusedIndex] if isFocusedBlock else None, focusedBlockRank)
            beta = localComm.bcast(beta, focusedBlockRank)

            v = v / scalingFactor

        return v, beta


    def fill(self):
        if self.index is not None:
            self.data[:,:] = np.random.randint(low=0, high=10, size=self.data.shape)
        self.comm.barrier()

    # TODO: make sure that it is copied to the same cores
    def copy(self):
        copiedMatrix = BlockMatrix(self.rowPartitions, self.columnPartitions, self.comm)

        if self.index is not None:
            self.comm.isend(self.data, copiedMatrix.blocks[self.index])

        if copiedMatrix.index is not None:
            copiedMatrix.data = self.comm.recv(source=self.blocks[copiedMatrix.index])

        return copiedMatrix

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