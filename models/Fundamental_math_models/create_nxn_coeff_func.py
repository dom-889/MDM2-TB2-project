'''
square of size N x N coefficients matrix
'''
import numpy as np
n = 5

#Make a diagonal of 1s for each diagonal
def create_para_diag_matrix(n, offset):
    matrix = np.zeros((n,n))
    if offset >= 0:
        for i in range(n - offset):
            matrix[i][i + offset] = 1
    else:
        for i in range(-offset, n):
            matrix[i][i + offset] = 1
    return matrix  

def create_nxn_coeff_matrix(n):
    A = np.zeros((6*n - 6, n**2))
    #rows
    for i in range(n):
        for j in range(n):
            A[i][i*n + j] = 1
    #columns
    for i in range(n):
        for j in range(n):
            A[n + j][i*n + j] = 1
    #Top left to bottom right diagonals
    for j in range(-n + 2, n-1):
        A[3*n + j-2] = create_para_diag_matrix(n, j).flatten()
    #Top right to bottom left diagonals
    for i in range(-n + 2, n-1):
        A[5*n -5 + i] = np.fliplr(create_para_diag_matrix(n, i)).flatten()
    
    return A
'''
#row sums
for i in range(n):
    for j in range(n):
        A[i][i*n + j] = 1

#column sums
for i in range(n):
    for j in range(n):
        A[n + j][i*n + j] = 1

#Make a diagonal of 1s for each diagonal
def create_para_diag_matrix(n, offset):
    matrix = np.zeros((n,n))
    if offset >= 0:
        for i in range(n - offset):
            matrix[i][i + offset] = 1
    else:
        for i in range(-offset, n):
            matrix[i][i + offset] = 1
    return matrix  

#Top left to bottom right diagonals
for j in range(-n + 2, n-1):
        A[3*n + j-2] = create_para_diag_matrix(n, j).flatten()

#Top right to bottom left diagonals
for i in range(-n + 2, n-1):
    A[5*n -5 + i] = np.fliplr(create_para_diag_matrix(n, i)).flatten()
'''
#print(A)
print(create_nxn_coeff_matrix(4))