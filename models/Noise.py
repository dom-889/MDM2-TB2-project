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
A, b_clean, img = ring_thing(fan_list, 
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

def get_edge_sharpness(image, global_min, global_max, threshold_ratio=0.15):
    """Measures gradient magnitude with noise suppression and thresholding."""
    scaled_image = (image - global_min) / (global_max - global_min + 1e-8)
    blurred = cv.GaussianBlur(scaled_image.astype(np.float32), (5, 5), 0)
    grad_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    max_edge = np.max(magnitude)
    clean_magnitude = np.where(magnitude > (max_edge * threshold_ratio), magnitude, 0)
    return np.mean(clean_magnitude), clean_magnitude

# Calculate Global Scale
g_min, g_max = np.min(true_img), np.max(true_img)
s_true, _ = get_edge_sharpness(true_img, g_min, g_max)
data_range = g_max - g_min

# ---------------------------------------------------------
# 3. NOISE SENSITIVITY ANALYSIS LOOP
# ---------------------------------------------------------
print("\n--- Starting Noise Sensitivity Analysis ---")

noise_levels = [0, 0.02, 0.05, 0.15] # 0%, 2%, 5%, 15% noise
sharpness_scores = []
ssim_scores = []
reconstructed_images = []
edge_maps = []

fixed_iterations = 20

for level in noise_levels:
    print(f"Processing Noise Level: {level*100}%...")
    
    # Add Gaussian Noise to Projection Data (Sinogram space)
    std_dev = level * np.mean(np.abs(b_clean))
    noise = np.random.normal(0, std_dev, b_clean.shape)
    b_noisy = b_clean + noise
    
    # Run Reconstruction
    x_recon = ART_solver(A, b_noisy, num_iterations=fixed_iterations)
    recon_img = x_recon.reshape(N, N)
    
    # Optional Denoising for metric stability
    clean_recon = cv.medianBlur(recon_img.astype(np.float32), 3)
    
    # Calculate Metrics
    s_recon, e_map = get_edge_sharpness(clean_recon, g_min, g_max)
    sharpness_scores.append((s_recon / s_true) * 100)
    ssim_scores.append(ssim(true_img, clean_recon, data_range=data_range) * 100)
    
    # Store for plotting
    reconstructed_images.append(recon_img)
    edge_maps.append(e_map)

# ---------------------------------------------------------
# 4. VISUAL COMPARISON GRID (The "Full" Comparison)
# ---------------------------------------------------------
fig, axes = plt.subplots(2, len(noise_levels) + 1, figsize=(18, 8))

# Column 0: Ground Truth
axes[0, 0].imshow(true_img, cmap='gray')
axes[0, 0].set_title("GROUND TRUTH\n(Original)", fontweight='bold')
_, gt_edges = get_edge_sharpness(true_img, g_min, g_max)
axes[1, 0].imshow(gt_edges, cmap='magma')
axes[1, 0].set_title("GT Edge Map")

for i, level in enumerate(noise_levels):
    col = i + 1
    # Top Row: Reconstructions
    axes[0, col].imshow(reconstructed_images[i], cmap='gray')
    axes[0, col].set_title(f"Noise: {level*100}%\nSSIM: {ssim_scores[i]:.1f}%")
    
    # Bottom Row: Edge Maps (Convolutional Layer)
    axes[1, col].imshow(edge_maps[i], cmap='magma')
    axes[1, col].set_title(f"Edges ({sharpness_scores[i]:.1f}%)")

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.savefig("images/results/noise_visual_comparison.png", dpi=150)
plt.show(block=False)
plt.pause(2)

# ---------------------------------------------------------
# 5. QUANTITATIVE GRAPH
# ---------------------------------------------------------
fig2, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Noise Level Added to Projections (%)', fontweight='bold')
ax1.set_ylabel('Sharpness Preservation (%)', color=color, fontweight='bold')
line1 = ax1.plot([n*100 for n in noise_levels], sharpness_scores, marker='o', color=color, linewidth=2.5, label='Edge Sharpness')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Structural Similarity - SSIM (%)', color=color, fontweight='bold')
line2 = ax2.plot([n*100 for n in noise_levels], ssim_scores, marker='s', color=color, linewidth=2.5, label='SSIM')
ax2.tick_params(axis='y', labelcolor=color)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower left')
plt.title("Quantitative Impact of Noise on ART Reconstruction", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("images/results/noise_graph_analysis.png", dpi=150)
plt.show()