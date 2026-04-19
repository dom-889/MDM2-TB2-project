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
phantom[:, :]         = 180  # Background
phantom[15:50, 15:50] = 100  # Soft Tissue
phantom[25:40, 25:40] = 30   # Bone / Dense Region
cv.imwrite("test_images/phantom.png", phantom)

# ---------------------------------------------------------
# 2. FORWARD PROJECTION  (Nr=180, Nb=96 → 11,520 total rays)
# ---------------------------------------------------------
fan_list = fan_setup(np.pi/4, no_beams=96)
A, b, img = ring_thing(fan_list,
                       ring_subdivisions=180,
                       beam_subdivisions=100,
                       aperture=1,
                       image_string="phantom.png",
                       resize=N)

# Ground truth in log-space
x_true   = np.log(np.clip(
    cv.cvtColor(cv.resize(cv.imread("test_images/phantom.png"), (N, N)),
                cv.COLOR_BGR2GRAY).astype(float) / 255, 1e-6, None)).flatten()
true_img = x_true.reshape(N, N)

g_min      = np.min(true_img)
g_max      = np.max(true_img)
data_range = g_max - g_min

# ---------------------------------------------------------
# METRIC HELPERS
# ---------------------------------------------------------
def get_edge_sharpness(image, global_min, global_max, threshold_ratio=0.15):
    scaled  = (image - global_min) / (global_max - global_min + 1e-8)
    blurred = cv.GaussianBlur(scaled.astype(np.float32), (5, 5), 0)
    grad_x  = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    grad_y  = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    clean     = np.where(magnitude > np.max(magnitude) * threshold_ratio, magnitude, 0)
    return np.mean(clean), clean

s_true, _ = get_edge_sharpness(true_img, g_min, g_max)

# ---------------------------------------------------------
# 3. NOISE LEVEL SWEEP
# ---------------------------------------------------------
print("\n--- Noise Level Sweep (Nr=180, 11,520 rays) ---")

noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
sharp_scores = []
ssim_scores  = []

for sigma in noise_levels:
    print(f"  Testing sigma = {sigma}...")

    np.random.seed(0)
    b_noisy = b + np.random.normal(0, sigma, size=b.shape)

    x_recon  = ART_solver(A, b_noisy, num_iterations=20)
    recon    = np.flipud(x_recon.reshape(N, N))
    clean    = cv.medianBlur(recon.astype(np.float32), 3)

    s_recon, _  = get_edge_sharpness(clean, g_min, g_max)
    sharp_scores.append((s_recon / s_true) * 100)
    ssim_scores.append(ssim(true_img, clean, data_range=data_range) * 100)

# ---------------------------------------------------------
# 4. PLOT: RELATIVE SHARPNESS vs SSIM ACROSS NOISE LEVELS
# ---------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 6.5))
fig.subplots_adjust(bottom=0.2)

color_sharp = 'tab:red'
color_ssim  = 'tab:blue'

ax1.set_xlabel('Noise Level (σ)', fontweight='bold')
ax1.set_ylabel('Sharpness Preservation (%)', color=color_sharp, fontweight='bold')
line1 = ax1.plot(noise_levels, sharp_scores, marker='o', linestyle='-',
                 color=color_sharp, linewidth=2.5, label='Sharpness Preservation')
ax1.tick_params(axis='y', labelcolor=color_sharp)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.axhline(y=100, color=color_sharp, linestyle=':', alpha=0.4, linewidth=1.5)

ax2 = ax1.twinx()
ax2.set_ylabel('Structural Similarity — SSIM (%)', color=color_ssim, fontweight='bold')
line2 = ax2.plot(noise_levels, ssim_scores, marker='s', linestyle='-',
                 color=color_ssim, linewidth=2.5, label='SSIM')
ax2.tick_params(axis='y', labelcolor=color_ssim)

# Clinical limit marker
ax1.axvline(x=0.1, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
ax1.text(0.12, ax1.get_ylim()[0], 'Clinical Limit',
         color='gray', fontsize=9, va='bottom')

plt.title("Noise Analysis (11,520 rays): Sharpness Preservation vs. SSIM",
          fontsize=14, fontweight='bold')

lines  = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
           fancybox=True, shadow=True, ncol=2, fontsize=10, frameon=True)

fig.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("project/Images/noise_analysis_dual_metric.png", dpi=150)
plt.show()

print("\nSaved to project/Images/noise_analysis_dual_metric.png")