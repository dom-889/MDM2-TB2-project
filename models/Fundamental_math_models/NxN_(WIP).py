'''
square matrix of size N x N
'''
import numpy as np
n = 3
A = np.zeros((n**2 + n, n**2))
#row sums
for i in range(n):
    for j in range(n):
        A[i][i*n + j] = 1

#column sums
for i in range(n):
    for j in range(n):
        A[n + j][i*n + j] = 1

#diagonal sums
for i in range(n):
    A[2*n + 1] = np.eye(n).flatten()



print(A)