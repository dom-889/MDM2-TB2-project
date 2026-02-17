'''
Some code for the fundamental maths Tony was on about on Monday
Takes the product of each row, column and diagonal, then logs it, eg. log(a) + log(b) = log(Product of first row)
This then gives you a system of equations (6 equations, 4 unknowns), although coeff matrix isn't square so can't be inversed
Use some teccy python function to get a pseudo-inverse, which gives a "good enough" inverse A to solve system of equations
'''

import numpy as np

#Changeable values how much "energy" each "pixel" on the body has
a = 0.2
b = 0.4
c = 0.6
d = 0.8

possible_values = [a, b, c, d]
letters = ['a', 'b', 'c', 'd']

#Very basic model of body with 4 pixels, can be changed around for model to guess any combination
matrix = np.array([[a, b],
                   [c, d]])

#Matrix of coefficients
A = np.array([[1, 1, 0, 0],
              [0, 0, 1, 1],
              [1, 0, 1, 0],
              [0, 1, 0, 1],
              [1, 0, 0, 1],
              [0, 1, 1, 0]])

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
x = A_pseudo_inv.dot(b)

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