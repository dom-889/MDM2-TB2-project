import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


def FBP_solver(A, b, img_shape, num_iterations=20, lam=1.0):
    """
    Iterative Filtered Backprojection solver.
    Solves Ax = b using:
        x^(k+1) = x^(k) + lambda * (A^T * F^-1{|w| * F{b - Ax^(k)}}) / sum_i(A_ij)
    
    Parameters:
        A              : path matrix (R x M)
        b              : measurement vector (R,)
        img_shape      : tuple (N, N) for reshaping output
        num_iterations : number of iterations
        lam            : relaxation parameter (default 1.0)
    
    Returns:
        x : reconstructed image vector (M,)
    """
    M = A.shape[1]
    x = np.zeros(M)  # initial guess

    # precompute normalisation: sum of each column of A
    # this is how many rays hit each pixel
    col_sums = np.sum(A, axis=0)
    col_sums[col_sums == 0] = 1  # avoid division by zero for unvisited pixels

    for iteration in range(num_iterations):

        # compute residual in projection space 
        residual = b - A @ x  # shape (R,)

        # apply ramp filter to each residual projection 
        # the ramp filter amplifies high frequencies: |w| in frequency space
        filtered_residual = np.zeros_like(residual)
        for i in range(len(residual)):
            # treat each residual as a 1D signal and apply ramp filter
            # here we apply it ray by ray
            freq = np.fft.fft(residual[i:i+1])
            freqs = np.fft.fftfreq(1)
            ramp = np.abs(freqs)
            filtered = np.fft.ifft(ramp * freq).real
            filtered_residual[i] = filtered[0]

        # backproject filtered residuals ---
        # A^T maps from projection space back to image space
        backprojected = A.T @ filtered_residual  # shape (M,)

        # normalise and update ---
        x = x + lam * (backprojected / col_sums)

        print(f"FBP iteration {iteration+1} of {num_iterations} complete")

    return x


def FBP_solver_vectorised(A, b, img_shape, num_iterations=20, lam=1.0):
    """
    More efficient vectorised FBP solver.
    Applies the ramp filter to the full residual vector at once.
    
    Parameters:
        A              : path matrix (R x M)
        b              : measurement vector (R,)
        img_shape      : tuple (N, N) for reshaping output
        num_iterations : number of iterations
        lam            : relaxation parameter (default 1.0)
    
    Returns:
        x : reconstructed image vector (M,)
    """
    M = A.shape[1]
    R = A.shape[0]
    x = np.zeros(M)

    # precompute normalisation
    col_sums = np.sum(A, axis=0)
    col_sums[col_sums == 0] = 1

    for iteration in range(num_iterations):

        # residual in projection space 
        residual = b - A @ x  # shape (R,)

        # apply ramp filter to full residual vector 
        freq_residual = np.fft.fft(residual)                    # FFT of residual
        freqs = np.fft.fftfreq(R)                               # frequency bins
        ramp = np.abs(freqs)                                     # ramp filter |w|
        filtered_residual = np.fft.ifft(ramp * freq_residual).real  # back to real space

        # backproject and normalise 
        backprojected = A.T @ filtered_residual
        x = x + lam * (backprojected / col_sums)

        print(f"FBP iteration {iteration+1} of {num_iterations} complete")

    return x


if __name__ == "__main__":
    # import ring_thing and fan_setup from your main model file 

    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from fixed_model import fan_setup, ring_thing, ART_solver

    os.makedirs("test_images", exist_ok=True)
    os.makedirs("project/Images", exist_ok=True)

    # create synthetic phantom 
    N = 64
    phantom = np.zeros((N, N, 3), dtype=np.uint8)
    phantom[:, :] = 180                         # background
    phantom[15:50, 15:50] = 100                 # soft tissue region
    phantom[25:40, 25:40] = 30                  # dense region (bone-like)
    cv.imwrite("test_images/phantom.png", phantom)
    print("Phantom created")

    # build A and b 
    fan_list = fan_setup(np.pi/4, no_beams=64)
    A, b, img = ring_thing(fan_list,
                           ring_subdivisions=90,
                           beam_subdivisions=100,
                           aperture=1,
                           image_string="phantom.png",
                           resize=N)

    # run ART 
    print("\nRunning ART...")
    x_art = ART_solver(A, b, num_iterations=20)
    recon_art = x_art.reshape(N, N)

    # run FBP 
    print("\nRunning FBP...")
    x_fbp = FBP_solver_vectorised(A, b, img_shape=(N, N), num_iterations=20)
    recon_fbp = x_fbp.reshape(N, N)

    # ground truth 
    x_true = np.log10(np.clip(cv.cvtColor(
        cv.resize(cv.imread("test_images/phantom.png"), (N, N)),
        cv.COLOR_BGR2GRAY).astype(float) / 255, 1e-6, None)).flatten()
    true_img = x_true.reshape(N, N)

    # compute RMSE 
    def rmse(a, b):
        return np.sqrt(np.mean((a - b)**2))

    rmse_art = rmse(x_art, x_true)
    rmse_fbp = rmse(x_fbp, x_true)
    print(f"\nRMSE ART: {rmse_art:.4f}")
    print(f"RMSE FBP: {rmse_fbp:.4f}")

  # Figure 1: reconstructions side by side 
    fig1, axes1 = plt.subplots(1, 3, figsize=(12, 4))
 
    axes1[0].imshow(true_img, cmap='gray')
    axes1[0].set_title("Original phantom")
    axes1[0].axis('off')
 
    axes1[1].imshow(recon_art, cmap='gray')
    axes1[1].set_title(f"ART\nRMSE = {rmse_art:.4f}")
    axes1[1].axis('off')
 
    axes1[2].imshow(recon_fbp, cmap='gray')
    axes1[2].set_title(f"FBP\nRMSE = {rmse_fbp:.4f}")
    axes1[2].axis('off')
 
    fig1.suptitle("ART vs FBP Reconstruction Comparison")
    fig1.tight_layout()
    fig1.savefig("images/results/ART_vs_FBP_reconstructions.png", dpi=150, bbox_inches='tight')
    plt.show()
 
    # Figure 2: difference map with colorbar
    fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
 
    diff = np.abs(recon_art - recon_fbp)
    im = ax2.imshow(diff, cmap='hot')
    ax2.set_title("Pixel-wise Absolute Error\n|ART - FBP|")
    ax2.axis('off')
 
    cbar = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute error", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
 
    fig2.tight_layout()
    fig2.savefig("images/results/ART_vs_FBP_difference.png", dpi=150, bbox_inches='tight')
    plt.show()
 