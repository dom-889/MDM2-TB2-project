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
phantom[:, :] = 180          # Background
phantom[15:50, 15:50] = 100  # Soft Tissue
phantom[25:40, 25:40] = 30   # Bone/Dense Region
cv.imwrite("test_images/phantom.png", phantom)

# Prepare Ground Truth (Log-space) for SSIM comparison
x_true = np.log10(np.clip(
    cv.cvtColor(cv.resize(cv.imread("test_images/phantom.png"), (N, N)), 
                cv.COLOR_BGR2GRAY).astype(float) / 255, 1e-6, None)).flatten()
true_img = x_true.reshape(N, N)

# Calculate Global Scale for the metrics
g_min = np.min(true_img)
g_max = np.max(true_img)
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
    
    # Filter out the background ringing artifacts
    max_edge = np.max(magnitude)
    clean_magnitude = np.where(magnitude > (max_edge * threshold_ratio), magnitude, 0)
    
    return np.mean(clean_magnitude), clean_magnitude

# Get baseline ground truth sharpness for the loop
s_true, _ = get_edge_sharpness(true_img, g_min, g_max)

# ---------------------------------------------------------
# 3. AUTOMATED BEAM ANALYSIS (DOSE REDUCTION)
# ---------------------------------------------------------
print("\n--- Starting Beam (Dose) Analysis ---")

# We test from extremely low dose (8 beams) up to high dose (128 beams)
beam_tests = [8, 16, 32, 48, 64, 96, 128]

sharpness_scores = []
ssim_scores = []

for beams in beam_tests:
    print(f"\n--- Testing Geometry with {beams} beams ---")
    
    # 1. Rebuild the physics geometry for this specific number of beams
    current_fan_list = fan_setup(np.pi/4, no_beams=beams)
    
    # 2. Re-simulate the forward projection 
    A_current, b_current, _ = ring_thing(current_fan_list, 
                                         ring_subdivisions=90,  # Keeping projection angles locked
                                         beam_subdivisions=100, 
                                         aperture=1, 
                                         image_string="phantom.png", 
                                         resize=N)
    
    # 3. Run reconstruction 
    # Locked at 10 iterations based on our previous efficiency analysis
    print(f"Solving with optimal 10 iterations...")
    x_recon = ART_solver(A_current, b_current, num_iterations=10)
    temp_recon_img = x_recon.reshape(N, N)
    
    # 4. Apply Denoising Filter (Regularization)
    clean_recon = cv.medianBlur(temp_recon_img.astype(np.float32), 3)
    
    # 5. Calculate Metrics
    s_recon, _ = get_edge_sharpness(clean_recon, g_min, g_max)
    preservation = (s_recon / s_true) * 100
    sharpness_scores.append(preservation)
    
    current_ssim = ssim(true_img, clean_recon, data_range=data_range)
    ssim_scores.append(current_ssim * 100)
    
    print(f"Result -> SSIM: {current_ssim*100:.1f}%, Sharpness: {preservation:.1f}%")

## ---------------------------------------------------------
# 4. OPTIMIZED PLOT FOR TECHNICAL REPORT
# ---------------------------------------------------------
# Smaller figsize makes the text look relatively larger when scaled
plt.rcParams.update({'font.size': 16}) # Global font increase
fig, ax1 = plt.subplots(figsize=(7, 5))

# Left Y-axis (Sharpness)
color = 'tab:red'
ax1.set_xlabel('Fan Beams ($N_b$)', fontweight='bold', fontsize=18)
ax1.set_ylabel('Sharpness (%)', color=color, fontweight='bold', fontsize=18)
line1 = ax1.plot(beam_tests, sharpness_scores, marker='o', color=color, 
                 linewidth=3, markersize=8, label='Sharpness')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

# Right Y-axis (SSIM)
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('SSIM (%)', color=color, fontweight='bold', fontsize=18)
line2 = ax2.plot(beam_tests, ssim_scores, marker='s', color=color, 
                 linewidth=3, markersize=8, label='SSIM')
ax2.tick_params(axis='y', labelcolor=color, labelsize=14)

# Title & Legend
plt.title("Quality vs. Beam Count ($N_b$)", fontsize=20, fontweight='bold', pad=15)
lines = line1 + line2
labels = [l.get_label() for l in lines]
# Small legend inside the plot area saves space
ax1.legend(lines, labels, loc='lower right', frameon=True, fontsize=12)

fig.tight_layout() # Removes unnecessary white margins
plt.savefig("project/Images/fanbeams_dense.png", dpi=300) # High DPI for crisp printing
plt.show()

# ---------------------------------------------------------
# 5. AUTOMATED RING SUBDIVISION ANALYSIS (ANGULAR SAMPLING)
# ---------------------------------------------------------
print("\n--- Starting Ring (Angular) Analysis ---")

# Standard test range for angular sampling
ring_sub_tests = [15, 30, 60, 90, 180, 360]
sharp_nr = []
ssim_nr = []

for rs in ring_sub_tests:
    print(f"Testing {rs} subdivisions...")
    
    # 1. Geometry with FIXED baseline beams (96)
    current_fan = fan_setup(np.pi/4, no_beams=96)
    
    # 2. Forward projection with varying ring subdivisions
    A_rs, b_rs, _ = ring_thing(current_fan, 
                               ring_subdivisions=rs, 
                               beam_subdivisions=100, 
                               aperture=1, 
                               image_string="phantom.png", 
                               resize=N)
    
    # 3. Reconstruction & Metrics
    x_rec = ART_solver(A_rs, b_rs, num_iterations=20) # Using optimal 20 iterations
    clean_rec = cv.medianBlur(x_rec.reshape(N,N).astype(np.float32), 3)
    
    s_rec, _ = get_edge_sharpness(clean_rec, g_min, g_max)
    sharp_nr.append((s_rec / s_true) * 100)
    ssim_nr.append(ssim(true_img, clean_rec, data_range=data_range) * 100)

# ---------------------------------------------------------
# 6. OPTIMIZED PLOT FOR Nr
# ---------------------------------------------------------
plt.rcParams.update({'font.size': 16})
fig, ax1 = plt.subplots(figsize=(7, 5))

# Left Y-axis (Sharpness)
color = 'tab:red'
ax1.set_xlabel('Ring Subdivisions ($N_r$)', fontweight='bold', fontsize=18)
ax1.set_ylabel('Sharpness (%)', color=color, fontweight='bold', fontsize=18)
ax1.plot(ring_sub_tests, sharp_nr, marker='o', color=color, linewidth=3, markersize=8, label='Sharpness')
ax1.tick_params(axis='y', labelcolor=color, labelsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)

# Right Y-axis (SSIM)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('SSIM (%)', color=color, fontweight='bold', fontsize=18)
ax2.plot(ring_sub_tests, ssim_nr, marker='s', color=color, linewidth=3, markersize=8, label='SSIM')
ax2.tick_params(axis='y', labelcolor=color, labelsize=14)

plt.title("Quality vs. Angular Sampling ($N_r$)", fontsize=20, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig("project/Images/ringsubdivisions_dense.png", dpi=300)
plt.show()