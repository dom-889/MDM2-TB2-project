'''
3 x 3 grid
'''


import numpy as np

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

#Very basic model of body with 4 pixels, can be changed around for model to guess any combination
matrix = np.array([[i, d, e],
                   [d, b, g],
                   [g, h, a]])

#Matrix of coefficients
A = np.array([[1,1,1,0,0,0,0,0,0],
              [0,0,0,1,1,1,0,0,0],
              [0,0,0,0,0,0,1,1,1],
              [1,0,0,1,0,0,1,0,0],
              [0,1,0,0,1,0,0,1,0],
              [0,0,1,0,0,1,0,0,1],
              [0,1,0,1,0,0,0,0,0],
              [0,0,1,0,1,0,1,0,0],
              [0,0,0,0,0,1,0,1,0],
              [0,0,0,1,0,0,0,1,0],
              [1,0,0,0,1,0,0,0,1],
              [0,1,0,0,0,1,0,0,0]])
             

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

print("Log values of a1, a2, b1, b2:", x)
print("Values of a1, a2, b1, b2:", solutions)
print("Projection:", rounded_letters.reshape(matrix.shape))