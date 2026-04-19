import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from fixed_model import fan_setup, ring_thing, ART_solver

# --- Generate the "Defect" Phantom ---
N = 64
defect_phantom = np.ones((N, N), dtype=np.uint8) * 180 # Uniform background (e.g., concrete/steel)
# A small, intense defect (e.g., fatigue crack/air void)
defect_phantom[28:36, 28:36] = 30  # Highly attenuating center
cv.imwrite("test_images/defect_phantom.png", defect_phantom)

# --- Simulate and Reconstruct with MDM4 ---
# Use the same parameters from your main analysis
fan_list = fan_setup(np.pi/4, no_beams=64)
A_def, b_def, img_def = ring_thing(fan_list, ring_subdivisions=180, beam_subdivisions=100, aperture=1, image_string="defect_phantom.png", resize=N)
print("Reconstructing Defect Phantom...")
x_def = ART_solver(A_def, b_def, num_iterations=30)
recon_def = np.flipud(x_def.reshape(N, N))

# --- Plot the results ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_def[:,:,0], cmap='gray')
axes[0].set_title("Ground Truth:\nIntact Structure with Defect")
axes[1].imshow(recon_def, cmap='gray')
axes[1].set_title("ART Reconstruction:\nAnomaly Localized for Targeted Repair")
for ax in axes: ax.axis('off')
plt.tight_layout()
plt.savefig("project/Images/ART_targeted_repair.png", dpi=100)
plt.show()
plt.show()