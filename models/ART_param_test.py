from fixed_model_new import fan_setup, ring_thing, ART_solver
#import os
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from skimage.metrics import structural_similarity as ssim

image_name = "shepp_logan_phantom.png"

n = 64

#Display image
phantom = cv.imread(f"test_images/{image_name}", cv.IMREAD_GRAYSCALE)
phantom = cv.resize(phantom, (n, n))

#ART RMSE comparisson image
x_true = np.log10(np.clip(cv.resize(cv.imread(f"test_images/{image_name}", cv.IMREAD_GRAYSCALE),(n, n)).astype(float) / 255, 1e-6, None)).flatten()
true_img = x_true.reshape(n, n)

g_min = np.min(true_img)
g_max = np.max(true_img)
data_range = g_max - g_min

def edge_rmse(ref, recon):
    sobel_ref_x = cv.Sobel(ref.astype(np.float32), cv.CV_64F, 1, 0, ksize=3)
    sobel_ref_y = cv.Sobel(ref.astype(np.float32), cv.CV_64F, 0, 1, ksize=3)
    edge_ref = np.sqrt(sobel_ref_x**2 + sobel_ref_y**2)

    sobel_recon_x = cv.Sobel(recon.astype(np.float32), cv.CV_64F, 1, 0, ksize=3)
    sobel_recon_y = cv.Sobel(recon.astype(np.float32), cv.CV_64F, 0, 1, ksize=3)
    edge_recon = np.sqrt(sobel_recon_x**2 + sobel_recon_y**2)

    edge_ref_norm = (edge_ref - np.min(edge_ref)) / (np.max(edge_ref) - np.min(edge_ref))
    edge_recon_norm = (edge_recon - np.min(edge_recon)) / (np.max(edge_recon) - np.min(edge_recon))

    rmse_edges = np.sqrt(np.mean((edge_ref_norm - edge_recon_norm)**2))
    return rmse_edges

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

s_true, _ = get_edge_sharpness(true_img, g_min, g_max)

# compute RMSE 
def compute_rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

#High resolution reconstruction for reference
fan_list_ref = fan_setup(np.pi/4, 256)
A_ref, b_ref, _ = ring_thing(fan_list_ref,
                            ring_subdivisions=360,
                            beam_subdivisions=48,
                            aperture=1,
                            image_string=image_name,
                            resize=n)
x_ref = ART_solver(A_ref, b_ref, num_iterations=50)
x_ref_corrected = np.flipud(x_ref.reshape(n, n)).flatten()
x_ref_corrected = x_ref_corrected.astype(float)
x_ref = (x_ref_corrected - np.min(x_ref_corrected)) / (np.max(x_ref_corrected) - np.min(x_ref_corrected))



ring_sub = 180
beam_size = 64
beam_subdivisions = [8, 16,32, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 78, 91, 104, 110, 119, 128]

fan_angle = np.pi/4
iterations = 20

beam_sub_rmses = []
beam_sub_edge_rmses = []

best_rmse = None
best_params = None

best_edge_rmse = None
best_edge_params = None

sharpness_scores = []
ssim_scores = []

print("Starting parameter sweep...")
fan_list = fan_setup(fan_angle, beam_size)
for beam_sub in beam_subdivisions:
    A, b, _ = ring_thing(fan_list,
                            ring_subdivisions=ring_sub,
                            beam_subdivisions=beam_sub,
                            aperture=1,
                            image_string=image_name,
                            resize=n)

    x = ART_solver(A, b, num_iterations=iterations)
    x_corrected = np.flipud(x.reshape(n, n)).flatten()
    x_corrected = x_corrected.astype(float)
    x_norm = (x_corrected - np.min(x_corrected)) / (np.max(x_corrected) - np.min(x_corrected))
    rmse = compute_rmse(x_norm, x_ref)
    edge_rmse_val = edge_rmse(x_norm.reshape(n, n), x_ref_corrected.reshape(n, n))

    temp_recon_img = x_corrected.reshape(n,n)
    clean_recon = cv.medianBlur(temp_recon_img.astype(np.float32), 3)
    s_recon, _ = get_edge_sharpness(clean_recon, g_min, g_max)
    preservation = (s_recon / s_true) * 100
    sharpness_scores.append(preservation)
    current_ssim = ssim(true_img, clean_recon, data_range=data_range)
    ssim_scores.append(current_ssim * 100)

    print(f"Ring: {ring_sub}, Beam: {beam_size}, Beam Sub: {beam_sub}, Iter: {iterations} -> RMSE: {rmse:.4f} Edge RMSE: {edge_rmse_val:.4f}")
    print(f"Fan Angle: {np.degrees(fan_angle):.1f}° -> RMSE: {rmse:.4f}")
    print(f'Edge Preservation: {preservation:.1f}%, SSIM: {current_ssim*100:.1f}%\n')
    if best_rmse is None or rmse < best_rmse:
        best_rmse = rmse
        best_params = {
                "ring_subdivisions": ring_sub,
                "beam_sizes": beam_size,
                "beam_subdivisions": beam_sub,
                "fan_angle": fan_angle,
                "iterations": iterations
                }

    if best_edge_rmse is None or edge_rmse_val < best_edge_rmse:
        best_edge_rmse = edge_rmse_val
        best_edge_params = {
            "ring_subdivisions": ring_sub,
            "beam_sizes": beam_size,
            "beam_subdivisions": beam_sub,
            "fan_angle": fan_angle,
            "iterations": iterations
        }
    beam_sub_rmses.append(rmse)
    beam_sub_edge_rmses.append(edge_rmse_val)
                   
fan_list_best = fan_setup(best_params["fan_angle"], best_params["beam_sizes"])
A_best, b_best, _ = ring_thing(fan_list_best,
                            ring_subdivisions=best_params["ring_subdivisions"],
                            beam_subdivisions=best_params["beam_subdivisions"],
                            aperture=1,
                            image_string=image_name,
                            resize=n)
x_best = ART_solver(A_best, b_best, num_iterations=best_params["iterations"])
x_best = np.flipud(x_best.reshape(n, n)).flatten()
x_best = x_best.astype(float)

edge_fan_list_best = fan_setup(best_edge_params["fan_angle"], best_edge_params["beam_sizes"])
A_edge_best, b_edge_best, _ = ring_thing(edge_fan_list_best,
                            ring_subdivisions=best_edge_params["ring_subdivisions"],
                            beam_subdivisions=best_edge_params["beam_subdivisions"],
                            aperture=1,
                            image_string=image_name,
                            resize=n)
x_edge_best = ART_solver(A_edge_best, b_edge_best, num_iterations=best_edge_params["iterations"])
x_edge_best = np.flipud(x_edge_best.reshape(n, n)).flatten()
x_edge_best = x_edge_best.astype(float)


x = np.array(beam_subdivisions)
y = np.array(beam_sub_rmses)
y_edge = np.array(beam_sub_edge_rmses)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(phantom, cmap='gray')
plt.title("Shepp-Logan Phantom")
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(x_best.reshape(n, n), cmap='gray')
plt.title(f"Reconstructed Image (RMSE: {best_rmse:.4f})")
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(x_edge_best.reshape(n, n), cmap='gray')
plt.title(f"Edge-Reconstructed Image (RMSE: {best_edge_rmse:.4f})")
plt.axis('off')
plt.tight_layout()


plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
#plt.plot(x, y, label='Data Points')
plt.axvline(x=32, color='black', linestyle='--', alpha=0.4)
plt.axvline(x=64, color='black', linestyle='--', alpha=0.4)
#plt.axvline(x=best_params['beam_subdivisions'], color='blue', linestyle='--', label=f"Best Beam Sub: {best_params['beam_subdivisions']}")
plt.plot(x, y_edge, label='Edge RMSE', color='orange')
plt.axvline(x=best_edge_params['beam_subdivisions'], color='orange', linestyle='--', label=f"Best Edge Beam Sub: {best_edge_params['beam_subdivisions']}")
plt.legend()
plt.grid(True)
plt.xlabel("Beam Subdivisions")
plt.ylabel("RMSE")
plt.title("Parameter Sweep")

plt.subplot(1,2,2)
#plt.plot(x, y, label='Data Points')
#plt.axvline(x=best_params['beam_subdivisions'], color='blue', linestyle='--', label=f"Best Beam Sub: {best_params['beam_subdivisions']}")
plt.plot(x, y_edge, label='Edge RMSE', color='orange')
plt.axvline(x=best_edge_params['beam_subdivisions'], color='orange', linestyle='--', label=f"Best Edge Beam Sub: {best_edge_params['beam_subdivisions']}")
plt.legend()
plt.xlim(40, 64)
#plt.ylim(0.09, 0.15)
plt.xlabel("Beam Subdivisions")
plt.ylabel("RMSE")
plt.title("Parameter Sweep (Zoomed)")
plt.grid(True)
plt.tight_layout()

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Number of Beam Subdivisions', fontweight='bold')
ax1.set_ylabel('Sharpness Preservation (%)', color=color, fontweight='bold')
line1 = ax1.plot(x, sharpness_scores, color=color, linewidth=2.5, label='Edge Sharpness')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Structural Similarity - SSIM (%)', color=color, fontweight='bold')
line2 = ax2.plot(x, ssim_scores, color=color, linewidth=2.5, label='SSIM')
ax2.tick_params(axis='y', labelcolor=color)

# Combine Legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right', frameon=True, shadow=True, borderpad=1)

plt.show()