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
        # TODO: make sure blocks of Q are stored in such a way that communication is minimized
        Q = BlockMatrix(self.rowPartitions, self.rowPartitions)
        Q.setIdentity()
        R = self.copy()

        for j in range(len(self.columnPartitions)):
            isFocusedColumn = self.index is not None and self.index[1] == j
            subCommunicator = self.comm.Split(0 if isFocusedColumn else MPI.UNDEFINED, self.index[0] if self.index is not None else MPI.UNDEFINED)
            
            W, Y = None, None

            if isFocusedColumn:
                W, Y = self.qrAux(R, j, subCommunicator)
            
            self.comm.barrier()

            self.applyWYToR(R, W, Y, j)
            self.applyWYToQ(Q, W, Y, j)

        self.comm.barrier()

        return Q.transpose(), R

    def applyWYToR(self, R, W, Y, columnPartitionIndex):
        # Send W and Y in seperate sends, since then Send instead of send can be used.
        # The advantage here is that Send doesn't require a pickle first and prevents problems
        # with too large objects
        WTag = 0
        YTag = 1

        if W is not None and Y is not None:
            for j in range(columnPartitionIndex + 1, len(self.columnPartitions)):
                self.comm.Isend(W, self.blocks[self.index[0], j], tag=WTag)
                self.comm.Isend(Y, self.blocks[self.index[0], j], tag=YTag)

        row = MPI.UNDEFINED if self.index is None else self.index[0]
        column = MPI.UNDEFINED if self.index is None else self.index[1]
        columnComm = self.comm.Split(column, row)

        if self.index is not None and self.index[1] > columnPartitionIndex:
            W = np.empty((self.data.shape[0], self.columnPartitions[columnPartitionIndex]))
            Y = np.empty((self.data.shape[0], self.columnPartitions[columnPartitionIndex]))
            self.comm.Recv(W, source=self.blocks[self.index[0], columnPartitionIndex], tag=WTag)
            self.comm.Recv(Y, source=self.blocks[self.index[0], columnPartitionIndex], tag=YTag)
            
            WRLocal = W.T @ R.data
            WR = columnComm.allreduce(WRLocal)
            R.data += Y @ WR

        self.comm.barrier()
    
    def applyWYToQ(self, Q, W, Y, columnPartitionIndex):
        # Send W and Y in seperate sends, since then Send instead of send can be used.
        # The advantage here is that Send doesn't require a pickle first and prevents problems
        # with too large objects
        WTag = 0
        YTag = 1

        if W is not None and Y is not None:
            for j in range(0, len(Q.columnPartitions)):
                self.comm.Isend(W, Q.blocks[self.index[0], j], tag=WTag)
                self.comm.Isend(Y, Q.blocks[self.index[0], j], tag=YTag)

        row = MPI.UNDEFINED if Q.index is None else Q.index[0]
        column = MPI.UNDEFINED if Q.index is None else Q.index[1]
        columnComm = Q.comm.Split(column, row)

        if Q.index is not None:
            W = np.empty((Q.data.shape[0], self.columnPartitions[columnPartitionIndex]))
            Y = np.empty((Q.data.shape[0], self.columnPartitions[columnPartitionIndex]))
            self.comm.Recv(W, source=self.blocks[Q.index[0], columnPartitionIndex], tag=WTag)
            self.comm.Recv(Y, source=self.blocks[Q.index[0], columnPartitionIndex], tag=YTag)
            
            WQLocal = W.T @ Q.data
            WQ = columnComm.allreduce(WQLocal)
            Q.data += Y @ WQ

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

    def transpose(self):
        B = BlockMatrix(self.columnPartitions, self.rowPartitions)

        if self.index is not None:
            self.comm.Isend(self.data, B.blocks[self.index[1], self.index[0]])

        if B.index is not None:
            buffer = np.empty((self.rowPartitions[B.index[1]], self.columnPartitions[B.index[0]]))
            self.comm.Recv(buffer, self.blocks[B.index[1], B.index[0]])
            B.data = buffer.T

        self.comm.barrier()

        return B

    def fill(self):
        if self.index is not None:
            self.data[:,:] = np.random.randint(low=0, high=10, size=self.data.shape)
        self.comm.barrier()

    def setIdentity(self):
        assert sum(self.rowPartitions) == sum(self.columnPartitions)

        if self.index is not None:
            row_from = sum(self.rowPartitions[:self.index[0]])
            row_to = row_from + self.rowPartitions[self.index[0]]
            column_from = sum(self.columnPartitions[:self.index[1]])
            column_to = column_from + self.columnPartitions[self.index[1]]

            self.data[:] = 0

            for row in range(row_from, row_to):
                for column in range(column_from, column_to):
                    if row != column:
                        continue

                    self.data[row - row_from, column - column_from] = 1

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
                self.comm.Isend(self.data, C.blocks[self.index[0], j], tag=self.id)

        if other.index is not None:
            for j in range(len(C.rowPartitions)):
                self.comm.Isend(other.data, C.blocks[j, other.index[1]], tag=other.id)

        if C.index is not None:
            for j in range(len(self.columnPartitions)):
                ABlock = np.empty((self.rowPartitions[C.index[0]], self.columnPartitions[j]))
                self.comm.Recv(ABlock, source=self.blocks[C.index[0], j], tag=self.id)
                BBlock = np.empty((other.rowPartitions[j], other.columnPartitions[C.index[1]]))
                self.comm.Recv(BBlock, source=other.blocks[j, C.index[1]], tag=other.id)
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
    
    def diag(self):
        if self.index is not None:
            row_from = sum(self.rowPartitions[:self.index[0]])
            row_to = row_from + self.rowPartitions[self.index[0]]
            column_from = sum(self.columnPartitions[:self.index[1]])
            column_to = column_from + self.columnPartitions[self.index[1]]

        diag = []

        for row in range(row_from, row_to):
            for column in range(column_from, column_to):
                if row != column:
                    continue

                diag.append(self.data[row - row_from, column - column_from])

        diag = self.comm.gather(np.array(diag))

        if self.comm.rank != 0:
            return None

        return diag