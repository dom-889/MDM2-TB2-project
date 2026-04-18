import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import sys
from skimage.metrics import structural_similarity as ssim
import matplotlib.transforms as mtransforms # Needed for precise legend placement

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fixed_model import fan_setup, ring_thing, ART_solver

# ---------------------------------------------------------
# 1. SETUP & METRICS
# ---------------------------------------------------------
N = 64
phantom = np.zeros((N, N, 3), dtype=np.uint8)
phantom[:, :] = 180
phantom[15:50, 15:50] = 100
phantom[25:40, 25:40] = 30
cv.imwrite("test_images/phantom.png", phantom)

x_true = np.log10(np.clip(
    cv.cvtColor(cv.resize(cv.imread("test_images/phantom.png"), (N, N)), 
                cv.COLOR_BGR2GRAY).astype(float) / 255, 1e-6, None)).flatten()
true_img = x_true.reshape(N, N)

g_min = np.min(true_img)
g_max = np.max(true_img)
data_range = g_max - g_min

def get_edge_sharpness(image, global_min, global_max, threshold_ratio=0.15):
    scaled_image = (image - global_min) / (global_max - global_min + 1e-8)
    blurred = cv.GaussianBlur(scaled_image.astype(np.float32), (5, 5), 0)
    grad_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    max_edge = np.max(magnitude)
    clean_magnitude = np.where(magnitude > (max_edge * threshold_ratio), magnitude, 0)
    return np.mean(clean_magnitude), clean_magnitude

s_true, _ = get_edge_sharpness(true_img, g_min, g_max)

# ---------------------------------------------------------
# 2. SWEEP 1: VARYING ANGLES (Fixed Beams = 64)
# ---------------------------------------------------------
print("Running Sweep 1: Varying Angles...")
angle_tests = [15, 30, 60, 90, 180, 360]
fixed_beams = 64
total_rays_angles = [a * fixed_beams for a in angle_tests]
ssim_angles, sharpness_angles = [], []

fixed_fan = fan_setup(np.pi/4, no_beams=fixed_beams)
for a in angle_tests:
    A, b, _ = ring_thing(fixed_fan, ring_subdivisions=a, beam_subdivisions=100, aperture=1, image_string="phantom.png", resize=N)
    x_recon = ART_solver(A, b, num_iterations=10).reshape(N, N)
    clean_recon = cv.medianBlur(x_recon.astype(np.float32), 3)
    
    ssim_angles.append(ssim(true_img, clean_recon, data_range=data_range) * 100)
    s_recon, _ = get_edge_sharpness(clean_recon, g_min, g_max)
    sharpness_angles.append((s_recon / s_true) * 100)

# ---------------------------------------------------------
# 3. SWEEP 2: VARYING BEAMS (Fixed Angles = 180)
# ---------------------------------------------------------
print("Running Sweep 2: Varying Beams...")
beam_tests = [8, 16, 32, 64, 128, 256]
fixed_angles = 180
total_rays_beams = [b * fixed_angles for b in beam_tests]
ssim_beams, sharpness_beams = [], []

for b in beam_tests:
    current_fan = fan_setup(np.pi/4, no_beams=b)
    A, b_vec, _ = ring_thing(current_fan, ring_subdivisions=fixed_angles, beam_subdivisions=100, aperture=1, image_string="phantom.png", resize=N)
    x_recon = ART_solver(A, b_vec, num_iterations=10).reshape(N, N)
    clean_recon = cv.medianBlur(x_recon.astype(np.float32), 3)
    
    ssim_beams.append(ssim(true_img, clean_recon, data_range=data_range) * 100)
    s_recon, _ = get_edge_sharpness(clean_recon, g_min, g_max)
    sharpness_beams.append((s_recon / s_true) * 100)
# ---------------------------------------------------------
# 4. PLOT THE MASTER COMPARISON (Preview Version)
# ---------------------------------------------------------

# Define the colors (Solid = Main, Dashed = Lighter)
color_angles_sharp = 'tab:red'         # Solid Sharpness
color_beams_sharp  = 'lightcoral'      # Lighter Dashed Sharpness
color_angles_ssim  = 'tab:blue'        # Solid SSIM
color_beams_ssim   = 'cornflowerblue'  # Lighter Dashed SSIM

fig, ax1 = plt.subplots(figsize=(11, 7.5))
fig.subplots_adjust(bottom=0.2) # Make room for legend at the bottom

# --- LEFT Y-AXIS: SHARPNESS (Reds) ---
ax1.set_xlabel('Total Number of Rays ($N_r \\times N_b$)', fontweight='bold')
ax1.set_ylabel('Sharpness Preservation (%)', color=color_angles_sharp, fontweight='bold')

line_sa = ax1.plot(total_rays_angles, sharpness_angles, marker='o', linestyle='-', 
                    color=color_angles_sharp, linewidth=2.5, label='Sharpness (Varying Angles - Solid)')
line_sb = ax1.plot(total_rays_beams, sharpness_beams, marker='s', linestyle='--', 
                    color=color_beams_sharp, linewidth=2.5, label='Sharpness (Varying Beams - Dashed)')

ax1.tick_params(axis='y', labelcolor=color_angles_sharp)
ax1.grid(True, linestyle='--', alpha=0.6)

# --- RIGHT Y-AXIS: SSIM (Blues) ---
ax2 = ax1.twinx()
ax2.set_ylabel('Structural Similarity - SSIM (%)', color=color_angles_ssim, fontweight='bold')

line_ia = ax2.plot(total_rays_angles, ssim_angles, marker='o', linestyle='-', 
                    color=color_angles_ssim, linewidth=2.5, label='SSIM (Varying Angles - Solid)')
line_ib = ax2.plot(total_rays_beams, ssim_beams, marker='s', linestyle='--', 
                    color=color_beams_ssim, linewidth=2.5, label='SSIM (Varying Beams - Dashed)')

ax2.tick_params(axis='y', labelcolor=color_angles_ssim)

# Vertical Line for Diminishing Returns (Exactly 11,520)
vline = ax1.axvline(x=11520, color='grey', linestyle=':', linewidth=2, label='Optimal Threshold (11,520 rays)')

plt.title('Reconstruction Quality vs. Total Simulated Rays', fontweight='bold', fontsize=14)

# --- LEGEND HANDLING ---
lines = line_sa + line_sb + line_ia + line_ib + [vline]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
           fancybox=True, shadow=True, ncol=2, fontsize=9, frameon=True)

fig.tight_layout(rect=[0, 0.15, 1, 1]) 

# Show the plot interactively!
plt.show()