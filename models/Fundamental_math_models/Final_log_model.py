'''
square of size N x N coefficients matrix
'''
import numpy as np
n = 4

#Changeable values how much "energy" each "pixel" on the body has
a = 0.1
b = 0.2
c = 0.3
d = 0.4
e = 0.5
f = 0.6
g = 0.7
h = 0.8
i = 0.9

possible_values = [a, b, c, d, e, f, g, h, i]
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

#Very basic model of body with 9 pixels, can be changed around for model to guess any combination
matrix = np.array([[i, d, e, f],
                   [d, b, g, a],
                   [g, h, a, e],
                   [f, a, e, c]])

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
A = create_nxn_coeff_matrix(n)

#Products of rows and columns logged
beta = np.zeros(A.shape[0])
for row in range(A.shape[0]):
    product  = 0
    for col in range(A.shape[1]):
        if A[row][col] == 1:
            product += np.log(matrix.flatten()[col])

    beta[row] = product
#A is singular so can't be inverted, use a pseudo-inverse instead
#Not great but good enough. Isn't always accurate
A_pseudo_inv = np.linalg.pinv(A)

#Solve matrix system of equations
x = A_pseudo_inv.dot(beta)

solutions = np.exp(x)

def round_to_closest(value, possible_values):
    return min(possible_values, key=lambda x: abs(x - value))

def value_to_letter(value, possible_values, letters):
    closest_value = round_to_closest(value, possible_values)
    return letters[possible_values.index(closest_value)]

rounded_letters = np.array([value_to_letter(val, possible_values, letters) for val in solutions])

print("Log values:", x.reshape(matrix.shape))
print("Values:", solutions.reshape(matrix.shape))
print("Projection:", rounded_letters.reshape(matrix.shape))