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
        isFirstColumn = self.index is not None and self.index[1] == 0
        subCommunicator = self.comm.Split(0 if isFirstColumn else MPI.UNDEFINED, self.index[0] if self.index is not None else MPI.UNDEFINED)

        if isFirstColumn:
            self.qrAux(subCommunicator)

        self.comm.barrier()

    def qrAux(self, localComm):
        R = self.data

        Y = np.zeros(self.data.shape)
        W = np.zeros(self.data.shape)
        
        for j in range(min(self.columnPartitions[0], sum(self.rowPartitions))):
            v, beta = self.householderReflectionAux(localComm, j)
            print(j, beta)
            if j == 0:
                Y[:,0] = v
                W[:,0] = -beta * v
            else:
                z = -beta * (v + W @ (Y.T @ v))
                Y[:,j] = v
                W[:,j] = z

            vR = v @ R
            K = localComm.allreduce(vR)
            R += -beta * (np.outer(v, K))

        return W, Y

    def householderReflection(self):
        isFirstColumn = self.index is not None and self.index[1] == 0
        subCommunicator = self.comm.Split(0 if isFirstColumn else MPI.UNDEFINED, self.index[0] if self.index is not None else MPI.UNDEFINED)

        if isFirstColumn:
            for j in range(min(self.columnPartitions[0], sum(self.rowPartitions))):
                self.householderReflectionAux(subCommunicator, j)

        self.comm.barrier()

    def householderReflectionAux(self, localComm, j):
        rowOffset = sum(self.rowPartitions[:self.index[0]])
        rowEnd = sum(self.rowPartitions[:self.index[0] + 1])

        focusedBlockRank = 0
        totalRowOffset = 0

        for i in range(len(self.rowPartitions) - 1):
            if j >= totalRowOffset and j < totalRowOffset + self.rowPartitions[i + 1]:
                focusedBlockRank = i

            totalRowOffset += self.rowPartitions[i]

        isFocusedBlock = j >= rowOffset and j < rowEnd
        localFocusedIndex = j - rowOffset if isFocusedBlock else None

        x = np.zeros(self.data.shape[0])
        if j < rowEnd:
            k = max(0, (j - rowOffset))
            x[k:] = self.data[k:,j]

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