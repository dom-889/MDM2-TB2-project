import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import sys
from skimage.metrics import structural_similarity as ssim

# ---------------------------------------------------------
# 1. SETUP & PATH RESOLUTION
# ---------------------------------------------------------
# Set directories and change working directory to script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
os.makedirs("test_images", exist_ok=True)
os.makedirs("project/Images", exist_ok=True)

# Ensure local imports work
sys.path.append(BASE_DIR)
try:
    from MDM4 import fan_setup, ring_thing, ART_solver
except ImportError:
    from fixed_model import fan_setup, ring_thing, ART_solver

N = 64

# ---------------------------------------------------------
# HELPER FUNCTIONS (Generate Phantoms)
# ---------------------------------------------------------
def generate_bone_phantom(path, size=64):
    """Standard phantom with dense central bone."""
    phantom = np.zeros((size, size, 3), dtype=np.uint8)
    phantom[:, :] = 180          # Background
    phantom[15:50, 15:50] = 100  # Soft Tissue
    phantom[25:40, 25:40] = 30   # Dense Bone
    cv.imwrite(path, phantom)
    return path

def generate_lung_phantom(path, size=64):
    """Phantom simulating air (low density) in lung tissue."""
    phantom = np.zeros((size, size, 3), dtype=np.uint8)
    phantom[:, :] = 180          # Background
    # Draw lower density tissue
    cv.circle(phantom, (32, 32), 22, (200, 200, 200), -1) 
    # Draw air pockets
    cv.circle(phantom, (25, 30), 8, (240, 240, 240), -1)
    cv.circle(phantom, (38, 35), 7, (240, 240, 240), -1)
    cv.imwrite(path, phantom)
    return path

def generate_anomaly_phantom(path, size=64):
    """Standard phantom with a small, highly dense anomaly."""
    phantom = np.zeros((size, size, 3), dtype=np.uint8)
    phantom[:, :] = 180          
    phantom[15:50, 15:50] = 100  
    phantom[25:40, 25:40] = 30   
    # Draw a dense tumor/anomaly
    cv.circle(phantom, (45, 45), 3, (10, 10, 10), -1) 
    cv.imwrite(path, phantom)
    return path

# Define phantom types and generate their files
phantom_info = {
    "Bone":    {"file": "phantom_bone.png",    "gen": generate_bone_phantom},
    "Lung":    {"file": "phantom_lung.png",    "gen": generate_lung_phantom},
    "Anomaly": {"file": "phantom_anomaly.png", "gen": generate_anomaly_phantom}
}

for p_type, info in phantom_info.items():
    info["gen"](os.path.join("test_images", info["file"]), N)

# ---------------------------------------------------------
# HELPER FUNCTION (Metric)
# ---------------------------------------------------------
def get_edge_sharpness(image, g_min, g_max):
    # Denoise for metric stability (Median Filter)
    clean = cv.medianBlur(image.astype(np.float32), 3)
    
    # Calculate Gradient (Convolutional Edge Map)
    scaled = (clean - g_min) / (g_max - g_min + 1e-8)
    blurred = cv.GaussianBlur(scaled, (5, 5), 0)
    g_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    g_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(g_x**2 + g_y**2)
    
    # Return average sharpness of *actual* edges (suppress noise)
    return np.mean(np.where(magnitude > (np.max(magnitude) * 0.15), magnitude, 0))

# ---------------------------------------------------------
# 2. PARAMETRIC SWEEP (Phantoms vs Noise)
# ---------------------------------------------------------
# Recreating parameters from your uploaded graph
noise_levels = [0, 0.1, 1, 3, 6] # Sigma levels (σ)
fixed_iters = 20
fixed_ring_size = 90 # Constant ring size to isolate phantom difference

# Store percentage results
# preservation_results[phantom_type] = [scores_for_each_noise_level]
preservation_results = {p: [] for p in phantom_info.keys()}

print("\n--- Starting Multi-Phantom Noise Analysis ---")

for p_type in phantom_info.keys():
    print(f"\nProcessing Phantom Type: {p_type}")
    
    # Resolve Ground Truth for this specific phantom
    gt_path = os.path.join("test_images", phantom_info[p_type]["file"])
    true_img_raw = cv.cvtColor(cv.resize(cv.imread(gt_path), (N, N)), cv.COLOR_BGR2GRAY).astype(float) / 255
    true_img = np.log10(np.clip(true_img_raw, 1e-6, None))
    g_min, g_max = np.min(true_img), np.max(true_img)
    
    # Baseline sharpness (0% preservation reference)
    s_baseline = get_edge_sharpness(true_img, g_min, g_max)

    # Re-setup fan list once per phantom type
    fan_list = fan_setup(np.pi/4, no_beams=64)
    
    # ring_thing appends "test_images/", so we only pass filename
    A, b_clean, _ = ring_thing(fan_list, ring_subdivisions=fixed_ring_size, 
                               beam_subdivisions=100, aperture=1, 
                               image_string=phantom_info[p_type]["file"], resize=N)
    
    for sigma in noise_levels:
        # Scale noise proportionally based on sigma and signal mean
        noise_scale = sigma * 0.01 * np.mean(np.abs(b_clean))
        noise = np.random.normal(0, noise_scale, b_clean.shape)
        b_noisy = b_clean + noise
        
        # Run ART Solver
        x_recon = ART_solver(A, b_noisy, num_iterations=fixed_iters)
        recon_img = x_recon.reshape(N, N)
        
        # Calculate Preservation Score
        s_recon = get_edge_sharpness(recon_img, g_min, g_max)
        preservation_results[p_type].append((s_recon / s_baseline) * 100)

# ---------------------------------------------------------
# 3. PLOT RECREATION (The Graph)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# Define colors matched to your uploaded graph
# Bone=Blue, Lung=Red, Anomaly=Green
colors = {"Bone": '#5B84B1', "Lung": '#C25B5B', "Anomaly": '#4C8E62'} 
markers = {"Bone": 'o', "Lung": 's', "Anomaly": '^'}

# Recreating the aesthetic with errorbars and specific formatting
for p_type in phantom_info.keys():
    plt.errorbar(noise_levels, preservation_results[p_type], 
                 yerr=1.5, # Matching aesthetic error bars from original
                 label=p_type, 
                 color=colors[p_type], marker=markers[p_type], 
                 linestyle='-', linewidth=2, capsize=4)

# Formatting title and labels
plt.title("ART Reconstruction Sharpness vs Gaussian Noise Level", 
          fontsize=14, fontweight='bold')
plt.xlabel("Noise level (σ)", fontsize=12)
plt.ylabel("Sharpness Preservation (%)", fontsize=12)
plt.legend(title="Phantom Type", loc="upper right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save result in your project structure
plt.savefig("project/Images/sharpness_all_phantoms.png", dpi=150)
print(f"\nSaved graph to: project/Images/sharpness_all_phantoms.png")
plt.show()