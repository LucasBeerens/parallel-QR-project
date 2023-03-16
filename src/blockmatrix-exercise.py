
total_number_of_cores = 20
machine = [Core(i) for i in range(total_number_of_cores)]

def distributed_block_matrix(name,row_partitions,column_partitions):
  block_matrices = {}

  for i in range(len(row_partitions)):
    for j in range(len(column_partitions)):
      core = machine[i * len(column_partitions) + j]
      core[name] = np.zeros((row_partitions[i], column_partitions[j]))
      core[f'{name}_shape'] = (row_partitions, column_partitions)
      core[f'{name}_index'] = (i, j)
      block_matrices[i, j] = core
    
  return block_matrices

def fill_block_matrix(A, name):
    for sub_matrix in A.values():
      shape = sub_matrix.memory[name].shape
      sub_matrix.memory[name] = np.random.randint(-5, 5, size=shape)

def apply_blockwise(A, name_A, B, name_B, target_name, operator):
  shape = A[0, 0][f'{name_A}_shape']
  C = distributed_block_matrix(target_name, shape[0], shape[1])
  
  for i, j in A:
    A[i, j].send(C[i, j], name_A)

  for i, j in B:
    B[i, j].send(C[i, j], name_B)

  for i, j in C:
    C[i, j].receive(A[i, j], name_A, f'{name_A}_tmp')
    C[i, j].receive(B[i, j], name_B, f'{name_B}_tmp')
    local_A = C[i, j][f'{name_A}_tmp']
    local_B = C[i, j][f'{name_B}_tmp']
    C[i, j][target_name] = operator(local_A, local_B)

    del C[i, j].memory[f'{name_A}_tmp']
    del C[i, j].memory[f'{name_B}_tmp']

  return C

# Adds two block matrices of the same structure
def add_block_matrix(A, name_A, B, name_B, target_name):
  return apply_blockwise(A, name_A, B, name_B, target_name, lambda x, y: x + y)





class BlockMatrix():
  def __init__(self, name, preallocated_distributed_block_matrix=None, row_partitions=None, column_partitions=None):
    self.name = name

    if preallocated_distributed_block_matrix is not None:
      self.distributed_block_matrix = preallocated_distributed_block_matrix
      shape = preallocated_distributed_block_matrix[0, 0][f'{name}_shape']
      self.row_partitions = shape[0]
      self.column_partitions = shape[1]
    elif row_partitions is not None and column_partitions is not None:
      self.distributed_block_matrix = distributed_block_matrix(name, row_partitions, column_partitions)
      self.row_partitions = row_partitions
      self.column_partitions = column_partitions
    else:
      raise Exception('Need to provide either a preallocated distributed block matrix or the row and column partitions')

  def fill(self):
    fill_block_matrix(self.distributed_block_matrix, self.name)

  def trace(self):
    return global_trace(self.distributed_block_matrix, self.name)

  @property
  def T(self):
    B = BlockMatrix(f'{self.name}.T', row_partitions=self.column_partitions, column_partitions=self.row_partitions)
    transpose_block_matrix(self.distributed_block_matrix, self.name, B.distributed_block_matrix, B.name)
    return B

  def __add__(self, other):
    sum_name = f'{self.name}+{other.name}'
    sum = add_block_matrix(self.distributed_block_matrix, self.name, other.distributed_block_matrix, other.name, sum_name)
    return BlockMatrix(sum_name, sum)

  def __neg__(self):
    neg = BlockMatrix(f'-{self.name}', row_partitions=self.row_partitions, column_partitions=self.column_partitions)

    for i, j in self.distributed_block_matrix:
      self.distributed_block_matrix[i, j].send(neg.distributed_block_matrix[i, j], self.name)

    for i, j in neg.distributed_block_matrix:
      tmp_name = f'{self.name}_tmp'
      neg.distributed_block_matrix[i, j].receive(self.distributed_block_matrix[i, j], self.name, tmp_name)
      neg.distributed_block_matrix[i, j][neg.name] = -neg.distributed_block_matrix[i, j][tmp_name]
      del neg.distributed_block_matrix[i, j].memory[tmp_name]

    return neg

  def __sub__(self, other):
    sub_name = f'{self.name}-{other.name}'
    sub = apply_blockwise(self.distributed_block_matrix, self.name, other.distributed_block_matrix, other.name, sub_name, lambda x, y: x - y)
    return BlockMatrix(sub_name, sub)

  def __mul__(self, other):
    mul_name = f'{self.name}+{other.name}'
    sub = apply_blockwise(self.distributed_block_matrix, self.name, other.distributed_block_matrix, other.name, mul_name, lambda x, y: x * y)
    return BlockMatrix(mul_name, sub)

  def __matmul__(self, other):
    target_name = f'{self.name}@{other.name}'
    prod_distributed_block_matrix = matrix_multiply_block_matrix(self.distributed_block_matrix, self.name, 
                                                                other.distributed_block_matrix, other.name,
                                                                target_name)
    return BlockMatrix(target_name, prod_distributed_block_matrix)

  def __str__(self):
    text = ''

    for i in range(len(self.row_partitions)):
      for row in range(self.row_partitions[i]):
        for j in range(len(self.column_partitions)):
          matrix_block = self.distributed_block_matrix[i, j][self.name]
          text += self.str_row_of_block(matrix_block, row)
        text += '\n'

    return text
      

  def str_row_of_block(self, matrix_block, row):
    text = ''
    
    for value in matrix_block[row, :]:
       text += f'{value: <6} '

    return text