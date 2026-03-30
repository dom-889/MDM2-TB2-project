import numpy as np

def FBP(A, b, num_iterations=10, lambda_val=1.0):
    """
    Solves Ax = b using iterative Filtered Back Projection (FBP):
        Applies a ramp filter at each iteration to refine the image estimate.
        x^(k+1) = x^(k) +  lambda_val * (A^T * inverse_fourier{filter * fourier{b - A * x^*(k)}}) / column_norms

    """
    M, P = A.shape # M: rays, P: pixels
    x = np.zeros(P)  # initial guess of all zeros

    col_norm = np.sum(A, axis=0)
    col_norm[col_norm == 0] = 1 #Avoid division by zero

    ramp = np.abs(np.fft.fftfreq(M)) #Ramp filter in the frequency domain

    for i in range(num_iterations):
        r = b - A @ x #Compute resdidual

        #Apply ramp filter in frequency domain
        r_fft = np.fft.fft(r,axis=0)
        r_fft_filtered = r_fft * ramp
        r_filtered = np.real(np.fft.ifft(r_fft_filtered, axis=0))


        delta_x = A.T @ r_filtered
        delta_x /= col_norm #Normalize by column sums

        x += lambda_val * delta_x #Update the image estimate

        print(f"FBP iteration {i+1} of {num_iterations} complete")

    return x