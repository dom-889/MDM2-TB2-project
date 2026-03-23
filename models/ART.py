import numpy as np

def ART_solver(A, b, num_iterations=10):

    """
    Solves Ax = b iteratively using the Kaczmarz (ART) update:
        x = x + ((b_i - a_i . x) / |a_i|^2) * a_i
    """
    M = A.shape[1] # number of pixels
    x = np.zeros(M)  # initial guess: all zeros

    # perform ART iterations
    # each iteration goes through all rays and updates the solution x
    for iteration in range(num_iterations):
        for i in range(len(b)):
            a_i = A[i]
            norm_sq = np.dot(a_i, a_i)
            if norm_sq == 0:
                continue  # skip empty rays
            residual = b[i] - np.dot(a_i, x)
            x = x + (residual / norm_sq) * a_i

        print(f"ART iteration {iteration+1} of {num_iterations} complete")

    return x
