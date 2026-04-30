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
# 1. SETUP & GROUND TRUTH PHANTOM
# ---------------------------------------------------------
os.makedirs("project/Images", exist_ok=True)
os.makedirs("test_images", exist_ok=True)
 
N = 64
phantom = np.zeros((N, N, 3), dtype=np.uint8)
phantom[:, :]         = 180  # Background
phantom[15:50, 15:50] = 100  # Soft Tissue
phantom[25:40, 25:40] = 30   # Bone/Dense Region
cv.imwrite("test_images/phantom.png", phantom)
 
# Prepare Ground Truth (Log-space) for SSIM comparison
x_true = np.log10(np.clip(
    cv.cvtColor(cv.resize(cv.imread("test_images/phantom.png"), (N, N)),
                cv.COLOR_BGR2GRAY).astype(float) / 255, 1e-6, None)).flatten()
true_img = x_true.reshape(N, N)
 
# Calculate Global Scale for the metrics
g_min      = np.min(true_img)
g_max      = np.max(true_img)
data_range = g_max - g_min
 
# ---------------------------------------------------------
# 2. METRICS FUNCTION
# ---------------------------------------------------------
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
 
# Get baseline ground truth sharpness
s_true, _ = get_edge_sharpness(true_img, g_min, g_max)

# ---------------------------------------------------------
# 3. BEAM COUNT SWEEP
# ---------------------------------------------------------
print("\n--- Starting Beam Count Analysis ---")

beam_tests = [8, 16, 32, 48, 64, 96, 128]

sharpness_scores = []
ssim_scores      = []

for beams in beam_tests:
    print(f"\n--- Testing {beams} beams ---")

    current_fan_list = fan_setup(np.pi/4, no_beams=beams)

    A_current, b_current, _ = ring_thing(current_fan_list,
                                          ring_subdivisions=180,
                                          beam_subdivisions=100,
                                          aperture=1,
                                          image_string="phantom.png",
                                          resize=N)

    x_recon        = ART_solver(A_current, b_current, num_iterations=20)
    temp_recon_img = np.flipud(x_recon.reshape(N, N))
    clean_recon    = cv.medianBlur(temp_recon_img.astype(np.float32), 3)

    s_recon, _   = get_edge_sharpness(clean_recon, g_min, g_max)
    preservation = (s_recon / s_true) * 100
    sharpness_scores.append(preservation)

    current_ssim = ssim(true_img, clean_recon, data_range=data_range)
    ssim_scores.append(current_ssim * 100)

    print(f"Result -> SSIM: {current_ssim*100:.1f}%, Sharpness: {preservation:.1f}%")

# ---------------------------------------------------------
# 4. PLOT
# ---------------------------------------------------------
plt.rcParams.update({'font.size': 16})
fig, ax1 = plt.subplots(figsize=(7, 5))

color = 'tab:red'
ax1.set_xlabel('Fan Beams ($N_b$)', fontweight='bold', fontsize=18)
ax1.set_ylabel('Sharpness (%)', color=color, fontweight='bold', fontsize=18)
line1 = ax1.plot(beam_tests, sharpness_scores, marker='o', color=color,
                 linewidth=3, markersize=8, label='Sharpness')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('SSIM (%)', color=color, fontweight='bold', fontsize=18)
line2 = ax2.plot(beam_tests, ssim_scores, marker='s', color=color,
                 linewidth=3, markersize=8, label='SSIM')
ax2.tick_params(axis='y', labelcolor=color, labelsize=14)

plt.title("Quality vs. Beam Count ($N_b$)", fontsize=20, fontweight='bold', pad=15)
lines  = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right', frameon=True, fontsize=12)

fig.tight_layout()
plt.savefig("project/Images/Nb1.png", dpi=300)
plt.show()