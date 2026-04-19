import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import sys
from skimage.metrics import structural_similarity as ssim

# Import your core logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fixed_model import fan_setup, ring_thing, ART_solver

def get_diagnostic_metrics(image, true_img):
    """Calculates Sharpness Preservation and SSIM."""
    # Normalize for metric consistency
    img_norm = image.astype(np.float32)
    true_norm = true_img.astype(np.float32)
    
    # 1. SSIM (Set data_range to 1.0 because we normalized to 0-1)
    score_ssim = ssim(true_norm.astype(np.float32), img_norm.astype(np.float32), data_range=1.0) * 100
    
    # 2. Sharpness (Sobel Magnitude)
    def compute_mag(im):
        gx = cv.Sobel(im, cv.CV_64F, 1, 0, ksize=3)
        gy = cv.Sobel(im, cv.CV_64F, 0, 1, ksize=3)
        return np.mean(np.sqrt(gx**2 + gy**2)), np.sqrt(gx**2 + gy**2)

    mag_true, _ = compute_mag(true_norm)
    mag_recon, edge_map = compute_mag(img_norm)
    preservation = (mag_recon / mag_true) * 100
    
    return score_ssim, preservation, edge_map

def FBP_solver_vectorised(A, b, img_shape, num_iterations=20, lam=1.0):
    M = A.shape[1]
    R = A.shape[0]
    x = np.zeros(M)
    col_sums = np.sum(A, axis=0)
    col_sums[col_sums == 0] = 1
    for iteration in range(num_iterations):
        residual = b - A @ x
        freq_residual = np.fft.fft(residual)
        freqs = np.fft.fftfreq(R)
        ramp = np.abs(freqs)
        filtered_residual = np.fft.ifft(ramp * freq_residual).real
        backprojected = A.T @ filtered_residual
        x = x + lam * (backprojected / col_sums)
    return x

if __name__ == "__main__":
    N = 64
    os.makedirs("project/Images", exist_ok=True)

    # 1. Build Data
    fan_list = fan_setup(np.pi/4, no_beams=64)
    A, b, img = ring_thing(fan_list, ring_subdivisions=180, beam_subdivisions=100, 
                           aperture=1, image_string="phantom.png", resize=N)

    # 2. Ground Truth Processing
    x_true = np.log10(np.clip(img[:,:,0].astype(float) / 255, 1e-6, None))
    true_img = x_true.reshape(N, N)

    # 3. Solve
    print("Running ART...")
    x_art = ART_solver(A, b, num_iterations=30)
    recon_art = np.flipud(x_art.reshape(N, N))

    print("Running FBP...")
    x_fbp = FBP_solver_vectorised(A, b, (N,N), num_iterations=20)
    recon_fbp = np.flipud(x_fbp.reshape(N, N))

    # 4. Metrics
    # --- ADD NORMALIZATION HERE ---
    def normalize_for_ssim(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

    recon_art_norm = normalize_for_ssim(recon_art)
    recon_fbp_norm = normalize_for_ssim(recon_fbp)
    true_img_norm = normalize_for_ssim(true_img)

    # Now run your metric function using the normalized versions
    # We update the function call to use the normalized images for SSIM
    ssim_art, sharp_art, edges_art = get_diagnostic_metrics(recon_art_norm, true_img_norm)
    ssim_fbp, sharp_fbp, edges_fbp = get_diagnostic_metrics(recon_fbp_norm, true_img_norm)
    # 4. Metrics
    ssim_art, sharp_art, edges_art = get_diagnostic_metrics(recon_art, true_img)
    ssim_fbp, sharp_fbp, edges_fbp = get_diagnostic_metrics(recon_fbp, true_img)

    # 5. Visual Comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# --- PLOTTING ---
    # Figsize (20, 8) makes the images significantly larger on screen
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), dpi=100)

    axes[0].imshow(true_img, cmap='gray')
    axes[0].set_title("Ground Truth", fontsize=18, fontweight='bold')
    
    axes[1].imshow(recon_art, cmap='gray')
    axes[1].set_title(f"ART Reconstruction\nSSIM: {ssim_art:.1f}% | Sharp: {sharp_art:.1f}%", 
                      fontsize=18, fontweight='bold')
    
    axes[2].imshow(recon_fbp, cmap='gray')
    axes[2].set_title(f"FBP Reconstruction\nSSIM: {ssim_fbp:.1f}% | Sharp: {sharp_fbp:.1f}%", 
                      fontsize=18, fontweight='bold')

    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("project/Images/Final_Comparison_Large.png", bbox_inches='tight')
    plt.show()