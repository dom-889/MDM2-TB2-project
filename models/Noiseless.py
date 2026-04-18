import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import sys
from skimage.metrics import structural_similarity as ssim

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fixed_model import fan_setup, ring_thing, ART_solver

# ---------------------------------------------------------
# 1. SETUP & PHANTOM GENERATION
# ---------------------------------------------------------
os.makedirs("project/Images", exist_ok=True)
os.makedirs("test_images", exist_ok=True)

N = 64
phantom = np.zeros((N, N, 3), dtype=np.uint8)
phantom[:, :] = 180          # Background
phantom[15:50, 15:50] = 100  # Soft Tissue
phantom[25:40, 25:40] = 30   # Bone/Dense Region
cv.imwrite("test_images/phantom.png", phantom)

# ---------------------------------------------------------
# 2. FORWARD PROJECTION (PHYSICS SIMULATION)
# ---------------------------------------------------------
fan_list = fan_setup(np.pi/4, no_beams=64)
A, b, img = ring_thing(fan_list, 
                       ring_subdivisions=90, 
                       beam_subdivisions=100, 
                       aperture=1, 
                       image_string="phantom.png", 
                       resize=N)

# Prepare Ground Truth (Log-space)
x_true = np.log10(np.clip(
    cv.cvtColor(cv.resize(cv.imread("test_images/phantom.png"), (N, N)), 
                cv.COLOR_BGR2GRAY).astype(float) / 255, 1e-6, None)).flatten()
true_img = x_true.reshape(N, N)

# ---------------------------------------------------------
# 3. BASELINE RECONSTRUCTION & METRICS
# ---------------------------------------------------------
print("\nReconstructing baseline using ART (Kaczmarz Method)...")
x_recon = ART_solver(A, b, num_iterations=20)
recon_img = x_recon.reshape(N, N)

def get_edge_sharpness(image, global_min, global_max, threshold_ratio=0.15):
    """Measures gradient magnitude with noise suppression and thresholding."""
    scaled_image = (image - global_min) / (global_max - global_min + 1e-8)
    blurred = cv.GaussianBlur(scaled_image.astype(np.float32), (5, 5), 0)
    
    grad_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Filter out the background ringing artifacts
    max_edge = np.max(magnitude)
    clean_magnitude = np.where(magnitude > (max_edge * threshold_ratio), magnitude, 0)
    
    return np.mean(clean_magnitude), clean_magnitude

# Calculate Global Scale
g_min = np.min(true_img)
g_max = np.max(true_img)

# Compute Baseline Sharpness Scores
s_true, _ = get_edge_sharpness(true_img, g_min, g_max)
s_recon, edge_map = get_edge_sharpness(recon_img, g_min, g_max)
preservation_score = (s_recon / s_true) * 100

print(f"\n--- DIAGNOSTIC QUALITY REPORT ---")
print(f"Ground Truth Sharpness: {s_true:.4f}")
print(f"Reconstructed Sharpness: {s_recon:.4f}")
print(f"Sharpness Preservation: {preservation_score:.2f}%")

# Plot Baseline
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(true_img, cmap='gray')
axes[0].set_title("Ground Truth\n(Input Phantom)")
axes[0].axis('off')

axes[1].imshow(recon_img, cmap='gray')
axes[1].set_title(f"ART Reconstruction\n(Preservation: {preservation_score:.1f}%)")
axes[1].axis('off')

axes[2].imshow(edge_map, cmap='magma')
axes[2].set_title("Convolutional Edge Map\n(Sobel Gradient)")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("project/Images/diagnostic_reconstruction.png", dpi=150)
plt.show(block=False) # Allows the script to continue to the loop without freezing
plt.pause(2)          # Pauses briefly to render the window

# ---------------------------------------------------------
# 4. AUTOMATED PARAMETER ANALYSIS
# ---------------------------------------------------------
print("\n--- Starting Parameter Analysis ---")

iteration_tests = [1, 2, 5, 10, 20, 30, 50]
sharpness_scores = []
ssim_scores = []

# Get baseline ground truth sharpness for the loop
s_true, _ = get_edge_sharpness(true_img, g_min, g_max)
data_range = g_max - g_min

for iters in iteration_tests:
    print(f"Testing ART with {iters} iterations...")
    
    # Run reconstruction
    x_recon = ART_solver(A, b, num_iterations=iters)
    temp_recon_img = x_recon.reshape(N, N)
    
    # Apply Denoising Filter (Total Variation proxy)
    clean_recon = cv.medianBlur(temp_recon_img.astype(np.float32), 3)
    
    # Calculate Sharpness (using cleaned image)
    s_recon, _ = get_edge_sharpness(clean_recon, g_min, g_max)
    sharpness_scores.append((s_recon / s_true) * 100)
    
    # Calculate SSIM (using cleaned image!)
    current_ssim = ssim(true_img, clean_recon, data_range=data_range)
    ssim_scores.append(current_ssim * 100)
    
# ---------------------------------------------------------
# 5. PLOT ANALYSIS RESULTS
# ---------------------------------------------------------
fig2, ax1 = plt.subplots(figsize=(10, 6))

# Left Y-axis (Sharpness)
color = 'tab:red'
ax1.set_xlabel('Number of ART Iterations', fontweight='bold')
ax1.set_ylabel('Sharpness Preservation (%)', color=color, fontweight='bold')
line1 = ax1.plot(iteration_tests, sharpness_scores, marker='o', color=color, linewidth=2.5, label='Edge Sharpness')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

# Right Y-axis (SSIM)
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Structural Similarity - SSIM (%)', color=color, fontweight='bold')
line2 = ax2.plot(iteration_tests, ssim_scores, marker='s', color=color, linewidth=2.5, label='SSIM')
ax2.tick_params(axis='y', labelcolor=color)

# Combine Legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=10)
plt.title("ART Iteration Analysis: Sharpness vs. SSIM", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("project/Images/iteration_analysis.png", dpi=150)
plt.show()

