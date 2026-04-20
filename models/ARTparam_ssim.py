from fixed_model_new import fan_setup, ring_thing, ART_solver
#import os
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from skimage.metrics import structural_similarity as ssim
import skimage.filters
from skimage.morphology import square

image_name = "shepp_logan_phantom.png"

n = 64

#Display image
phantom = cv.imread(f"test_images/{image_name}", cv.IMREAD_GRAYSCALE)
phantom = cv.resize(phantom, (n, n))

#ART high param reference imnage
fan_list_ref = fan_setup(np.pi/4, 128)
A_ref, b_ref, _ = ring_thing(fan_list_ref,
                            ring_subdivisions=360,
                            beam_subdivisions=61,
                            aperture=1,
                            image_string=image_name,
                            resize=n)
x_ref = ART_solver(A_ref, b_ref, num_iterations=30)
x_ref_corrected = np.flipud(x_ref.reshape(n, n)).flatten()
x_ref_norm = (x_ref_corrected - np.min(x_ref_corrected)) / (np.max(x_ref_corrected) - np.min(x_ref_corrected))
true_img = x_ref_norm.reshape(n, n)

g_min = np.min(true_img)
g_max = np.max(true_img)
data_range = g_max - g_min


def get_edge_sharpness(image, global_min, global_max, threshold_ratio=0.1):
    """Measures gradient magnitude with noise suppression and thresholding."""
    scaled_image = (image - global_min) / (global_max - global_min + 1e-8)
    blurred = cv.GaussianBlur(scaled_image.astype(np.float32), (5, 5), 0)
    
    grad_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    max_edge = np.max(magnitude)
    clean_magnitude = np.where(magnitude > (max_edge * threshold_ratio), magnitude, 0)
    
    return np.mean(clean_magnitude), clean_magnitude

s_true, edge_map_true = get_edge_sharpness(true_img, g_min, g_max, threshold_ratio=0)


ring_sub = 180
beam_size = 64
#beam_subdivisions = [16, 32, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 78, 91, 104, 110, 119, 128]
beam_subdivisions = np.linspace(10, 150, 50, dtype=int)

fan_angle = np.pi/4
iterations = 20


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
    x_corrected = (x_corrected - np.min(x_corrected)) / (np.max(x_corrected) - np.min(x_corrected))
    
    temp_recon_img = x_corrected.reshape(n,n)
    '''
    temp_uint8 = (temp_recon_img*255).astype(np.uint8)
    clean_recon = skimage.filters.median(temp_uint8, np.square(3)).astype(np.float32)/255
    '''
    clean_recon = cv.medianBlur(temp_recon_img.astype(np.float32), 3)
    s_recon, _ = get_edge_sharpness(clean_recon, g_min, g_max)#recon threshold here
    preservation = (s_recon / s_true) * 100
    sharpness_scores.append(preservation)
    current_ssim = ssim(true_img, clean_recon, data_range=data_range)
    ssim_scores.append(current_ssim * 100)

x = np.array(beam_subdivisions)

optimal_sub = beam_subdivisions[np.argmax(ssim_scores)]

A, b, _ = ring_thing(fan_list,
                            ring_subdivisions=ring_sub,
                            beam_subdivisions=optimal_sub,
                            aperture=1,
                            image_string=image_name,
                            resize=n)

x_optimal = ART_solver(A, b, num_iterations=iterations)
x_corrected = np.flipud(x_optimal.reshape(n, n)).flatten()
x_corrected = (x_corrected - np.min(x_corrected)) / (np.max(x_corrected) - np.min(x_corrected))
temp_recon_img = x_corrected.reshape(n,n)
clean_recon = cv.medianBlur(temp_recon_img.astype(np.float32), 3)
s_recon, edge_map_recon = get_edge_sharpness(clean_recon, g_min, g_max) #recon threshold here
preservation = (s_recon / s_true) * 100
current_ssim = ssim(true_img, clean_recon, data_range=data_range)
current_ssim *= 100

print(s_true)
print(s_recon)

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
#ax1.legend(lines, labels, loc='lower right', frameon=True, shadow=True, borderpad=1)
plt.title('ART Beam Subdivision Analysis (Shepp-Logan)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Shepp_Logan_results_final/ART_param_sweep.png', dpi=150)

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.imshow(true_img, cmap='gray')
plt.title('True Image')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(temp_recon_img, cmap='gray')
plt.title(f'Optimal Reconstruction\n{optimal_sub} Beam Subdivisions\nSSIM: {current_ssim:.2f}%  |  Edge Preservation: {preservation:.2f}%')
#plt.text(0.5, -0.1, f'SSIM: {current_ssim:.2f}%\nEdge Preservation: {preservation:.2f}%', ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
plt.axis('off')
plt.tight_layout()
plt.savefig('Shepp_Logan_results_final/ART_images.png', dpi=150)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(edge_map_true, cmap='hot')
plt.title('True Edge Map')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(edge_map_recon, cmap='hot', vmax=np.max(edge_map_true), vmin=0)
plt.title('Reconstructed Edge Map')
plt.axis('off')
plt.tight_layout()
plt.savefig('Shepp_Logan_results_final/ART_edge_heat_maps.png', dpi=150)

plt.show()
print(f'g_min: {g_min}, g_max: {g_max}, s_true: {s_true}, s_recon: {s_recon}')