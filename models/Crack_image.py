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
# 1. SETUP & CRACK PHANTOM GENERATION
# ---------------------------------------------------------
os.makedirs("project/Images", exist_ok=True)
os.makedirs("test_images", exist_ok=True)

SIZE = 256  # Generate at high resolution, downsample during projection
N    = 64   # Working resolution for ART

def generate_crack_phantom(size=256):
    """
    Generates a rectangular slab with branching cracks.

    Pixel values (0-255):
        230  - background (air, low attenuation)
        140  - dense slab material (concrete / steel)
        245  - crack (air gap, near-zero attenuation)
    """
    img = np.full((size, size), 230, dtype=np.uint8)

    # Rectangular slab
    x1, y1 = int(0.12 * size), int(0.15 * size)
    x2, y2 = int(0.88 * size), int(0.85 * size)
    img[y1:y2, x1:x2] = 140

    def draw_jagged_crack(image, start, end, steps=18, jitter=3, thickness=2, value=245):
        pts = [np.array(start, dtype=np.float32)]
        for i in range(1, steps):
            t = i / steps
            mid = (1 - t) * np.array(start) + t * np.array(end)
            mid += np.random.uniform(-jitter, jitter, 2)
            pts.append(mid)
        pts.append(np.array(end, dtype=np.float32))
        for i in range(len(pts) - 1):
            cv.line(image,
                    tuple(pts[i].astype(int)),
                    tuple(pts[i + 1].astype(int)),
                    int(value), thickness)

    np.random.seed(42)

    # Main crack (left to right through the slab)
    draw_jagged_crack(img,
                      (int(0.15 * size), int(0.48 * size)),
                      (int(0.78 * size), int(0.52 * size)),
                      steps=22, jitter=4, thickness=2)
    # Branch 1 - upward ~40% along main crack
    draw_jagged_crack(img,
                      (int(0.38 * size), int(0.49 * size)),
                      (int(0.58 * size), int(0.28 * size)),
                      steps=14, jitter=3, thickness=2)
    # Branch 2 - downward ~60% along main crack
    draw_jagged_crack(img,
                      (int(0.55 * size), int(0.51 * size)),
                      (int(0.72 * size), int(0.70 * size)),
                      steps=12, jitter=3, thickness=2)
    # Sub-branch off branch 1
    draw_jagged_crack(img,
                      (int(0.48 * size), int(0.38 * size)),
                      (int(0.60 * size), int(0.30 * size)),
                      steps=8, jitter=2, thickness=1)

    return img

phantom_256 = generate_crack_phantom(SIZE)
cv.imwrite("test_images/crack_phantom.png", phantom_256)
print("Crack phantom saved to test_images/crack_phantom.png")

# ---------------------------------------------------------
# 2. FORWARD PROJECTION (PHYSICS SIMULATION)
# Optimised: Nr=180, Nb=96 maximises SSIM for a 64x64 grid.
# More ring positions improve angular coverage more
# effectively than extra beams per fan.
# ---------------------------------------------------------
fan_list = fan_setup(np.pi/4, no_beams=96)
A, b, img = ring_thing(fan_list,
                       ring_subdivisions=180,
                       beam_subdivisions=100,
                       aperture=1,
                       image_string="crack_phantom.png",
                       resize=N)

# Prepare Ground Truth (Log-space)
x_true = np.log(np.clip(
    cv.cvtColor(cv.resize(cv.imread("test_images/crack_phantom.png"), (N, N)),
                cv.COLOR_BGR2GRAY).astype(float) / 255, 1e-6, None)).flatten()
true_img = x_true.reshape(N, N)

# ---------------------------------------------------------
# 3. BASELINE RECONSTRUCTION & METRICS
# ---------------------------------------------------------
print("\nReconstructing baseline using ART (Kaczmarz Method)...")
x_recon = ART_solver(A, b, num_iterations=20)

# Fix spatial inversion introduced by ring geometry
recon_img = np.flipud(x_recon.reshape(N, N))

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

# Global scale
g_min = np.min(true_img)
g_max = np.max(true_img)
data_range = g_max - g_min

# Compute baseline metrics
s_true, _          = get_edge_sharpness(true_img,  g_min, g_max)
s_recon, _         = get_edge_sharpness(recon_img, g_min, g_max)
preservation_score = (s_recon / s_true) * 100
baseline_ssim      = ssim(true_img, recon_img, data_range=data_range) * 100

print(f"\n--- DIAGNOSTIC QUALITY REPORT ---")
print(f"Ground Truth Sharpness : {s_true:.4f}")
print(f"Reconstructed Sharpness: {s_recon:.4f}")
print(f"Edge Preservation      : {preservation_score:.2f}%")
print(f"SSIM                   : {baseline_ssim:.2f}%")

# Two-panel plot (ground truth + reconstruction only)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(true_img, cmap='gray')
axes[0].set_title("Ground Truth\n(Crack Phantom)")
axes[0].axis('off')

axes[1].imshow(recon_img, cmap='gray')
axes[1].set_title(
    f"ART Reconstruction\n"
    f"SSIM: {baseline_ssim:.1f}%  |  Edge Preservation: {preservation_score:.1f}%"
)
axes[1].axis('off')

plt.tight_layout()
plt.savefig("project/Images/crack_diagnostic_reconstruction.png", dpi=150)
plt.show(block=False)
plt.pause(2)

# ---------------------------------------------------------
# 4. AUTOMATED PARAMETER ANALYSIS
# ---------------------------------------------------------
print("\n--- Starting Parameter Analysis ---")

iteration_tests  = [1, 2, 5, 10, 20, 30, 50]
sharpness_scores = []
ssim_scores      = []

s_true, _ = get_edge_sharpness(true_img, g_min, g_max)

for iters in iteration_tests:
    print(f"Testing ART with {iters} iterations...")

    x_recon        = ART_solver(A, b, num_iterations=iters)
    temp_recon_img = np.flipud(x_recon.reshape(N, N))

    # Median denoising before scoring
    clean_recon = cv.medianBlur(temp_recon_img.astype(np.float32), 3)

    s_recon, _ = get_edge_sharpness(clean_recon, g_min, g_max)
    sharpness_scores.append((s_recon / s_true) * 100)

    current_ssim = ssim(true_img, clean_recon, data_range=data_range)
    ssim_scores.append(current_ssim * 100)

# ---------------------------------------------------------
# 5. PLOT ANALYSIS RESULTS
# ---------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 6.5))
fig.subplots_adjust(bottom=0.2)

color_sharp = 'tab:red'
ax1.set_xlabel('Number of ART Iterations', fontweight='bold')
ax1.set_ylabel('Sharpness Preservation (%)', color=color_sharp, fontweight='bold')

line1 = ax1.plot(iteration_tests, sharpness_scores, marker='o', linestyle='-',
                 color=color_sharp, linewidth=2.5, label='Edge Sharpness')

ax1.tick_params(axis='y', labelcolor=color_sharp)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
color_ssim = 'tab:blue'
ax2.set_ylabel('Structural Similarity - SSIM (%)', color=color_ssim, fontweight='bold')

line2 = ax2.plot(iteration_tests, ssim_scores, marker='s', linestyle='-',
                 color=color_ssim, linewidth=2.5, label='SSIM')

ax2.tick_params(axis='y', labelcolor=color_ssim)

plt.title("ART Iteration Analysis - Crack Phantom: Sharpness vs. SSIM",
          fontsize=14, fontweight='bold')

lines  = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
           fancybox=True, shadow=True, ncol=2, fontsize=10, frameon=True)

fig.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("project/Images/crack_iteration_analysis.png", dpi=150)
plt.show()