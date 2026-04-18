import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import sys
from skimage.metrics import structural_similarity as ssim

# 1. SETUP & PATH RESOLUTION
# Ensure we are in the directory where the script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Ensure the "test_images" folder exists relative to this script
os.makedirs("test_images", exist_ok=True)
os.makedirs("project/Images", exist_ok=True)

# Important: ring_thing appends "test_images/" internally, so we only pass the filename
phantom_filename = "phantom.png" 
actual_file_path = os.path.join("test_images", phantom_filename)

# Ensure local imports work
sys.path.append(BASE_DIR)
try:
    from MDM4 import fan_setup, ring_thing, ART_solver
except ImportError:
    from fixed_model import fan_setup, ring_thing, ART_solver

N = 64

# Create the phantom if it doesn't exist
if not os.path.exists(actual_file_path):
    phantom = np.zeros((N, N, 3), dtype=np.uint8)
    phantom[:, :] = 180          
    phantom[15:50, 15:50] = 100  
    phantom[25:40, 25:40] = 30   
    cv.imwrite(actual_file_path, phantom)
    print(f"Generated phantom at: {actual_file_path}")

# Prepare Ground Truth for metrics
true_img_raw = cv.cvtColor(cv.resize(cv.imread(actual_file_path), (N, N)), cv.COLOR_BGR2GRAY).astype(float) / 255
true_img = np.log10(np.clip(true_img_raw, 1e-6, None))
g_min, g_max = np.min(true_img), np.max(true_img)

def get_metrics(recon, gt, g_min, g_max):
    clean = cv.medianBlur(recon.astype(np.float32), 3)
    scaled = (clean - g_min) / (g_max - g_min + 1e-8)
    blurred = cv.GaussianBlur(scaled, (5, 5), 0)
    grad_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x**2 + grad_y**2)
    s_score = np.mean(np.where(mag > (np.max(mag) * 0.15), mag, 0))
    s_sim = ssim(gt, clean, data_range=g_max - g_min)
    return s_score, s_sim

s_true, _ = get_metrics(true_img, true_img, g_min, g_max)

# 2. PARAMETRIC SWEEP
ring_configs = [30, 90, 180]
noise_levels = [0, 0.1, 1, 3, 6] 
fixed_iters = 20
results = {r: {"sharpness": [], "ssim": []} for r in ring_configs}

for r_size in ring_configs:
    print(f"\n--- Evaluating Ring Size: {r_size} ---")
    fan_list = fan_setup(np.pi/4, no_beams=64)
    
    # Passing ONLY the filename because ring_thing adds "test_images/"
    A, b_clean, _ = ring_thing(fan_list, ring_subdivisions=r_size, 
                               beam_subdivisions=100, aperture=1, 
                               image_string=phantom_filename, resize=N)
    
    for sigma in noise_levels:
        noise_scale = sigma * 0.01 * np.mean(np.abs(b_clean))
        noise = np.random.normal(0, noise_scale, b_clean.shape)
        b_noisy = b_clean + noise
        
        x_recon = ART_solver(A, b_noisy, num_iterations=fixed_iters)
        recon_img = x_recon.reshape(N, N)
        s_val, ssim_val = get_metrics(recon_img, true_img, g_min, g_max)
        
        results[r_size]["sharpness"].append((s_val / s_true) * 100)
        results[r_size]["ssim"].append(ssim_val * 100)

# 3. PLOTTING
plt.figure(figsize=(10, 6))
colors = {30: '#5B84B1', 90: '#4C8E62', 180: '#C25B5B'} 
markers = {30: 'o', 90: 's', 180: '^'}

for r_size in ring_configs:
    plt.errorbar(noise_levels, results[r_size]["sharpness"], yerr=1.2,
                 label=f"Ring = {r_size} ({r_size*64} rays)",
                 color=colors[r_size], marker=markers[r_size], linewidth=2, capsize=4)

plt.title("Noise Sensitivity vs Number of Rays (Sharpness Metric)", fontweight='bold')
plt.xlabel("Noise level (σ)")
plt.ylabel("Sharpness Preservation (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("project/Images/sharpness_vs_rays.png")
plt.show()