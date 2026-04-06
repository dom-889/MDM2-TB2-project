from fixed_model_new import fan_setup, ring_thing, ART_solver
#import os
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

image_name = "shepp_logan_phantom.png"

n = 64

#Display image
phantom = cv.imread(f"test_images/{image_name}", cv.IMREAD_GRAYSCALE)
phantom = cv.resize(phantom, (n, n))

#ART RMSE comparisson image
x_true = np.log10(np.clip(cv.resize(cv.imread(f"test_images/{image_name}", cv.IMREAD_GRAYSCALE),(n, n)).astype(float) / 255, 1e-6, None)).flatten()
true_img = x_true.reshape(n, n)



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



ring_subdivisions = [180]
beam_sizes = [64]
beam_subdivisions = [8, 16,32, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 78, 91, 104, 110, 119, 128]

fan_angles = [np.pi/4]
iterations = [20]

beam_sub_rmses = []

best_rmse = None
best_params = None

print("Starting parameter sweep...")
for ring_sub in ring_subdivisions:
    for beam_size in beam_sizes:
         for fan_angle in fan_angles:
            fan_list = fan_setup(fan_angle, beam_size)
            for beam_sub in beam_subdivisions:
                A, b, _ = ring_thing(fan_list,
                                        ring_subdivisions=ring_sub,
                                        beam_subdivisions=beam_sub,
                                        aperture=1,
                                        image_string=image_name,
                                        resize=n)
                for num_iter in iterations:
                    x = ART_solver(A, b, num_iterations=num_iter)
                    x_corrected = np.flipud(x.reshape(n, n)).flatten()
                    x_corrected = x_corrected.astype(float)
                    rmse = compute_rmse(x_corrected, x_ref_corrected)
                    rmse = rmse / (np.max(x_ref_corrected) - np.min(x_ref_corrected))  # Normalise RMSE

                    print(f"Ring: {ring_sub}, Beam: {beam_size}, Beam Sub: {beam_sub}, Iter: {num_iter} -> RMSE: {rmse:.4f}")
                    print(f"Fan Angle: {np.degrees(fan_angle):.1f}° -> RMSE: {rmse:.4f}")
                    if best_rmse is None or rmse < best_rmse:
                        best_rmse = rmse
                        best_params = {
                            "ring_subdivisions": ring_sub,
                            "beam_sizes": beam_size,
                            "beam_subdivisions": beam_sub,
                            "fan_angle": fan_angle,
                            "iterations": num_iter
                        }
                beam_sub_rmses.append(rmse)

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

x = np.array(beam_subdivisions)
y = np.array(beam_sub_rmses)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(phantom, cmap='gray')
plt.title("Shepp-Logan Phantom")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(x_best.reshape(n, n), cmap='gray')
plt.title(f"Reconstructed Image (RMSE: {best_rmse:.4f})")
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.plot(x, y, label='Data Points')
plt.axvline(x=32, color='black', linestyle='--', alpha=0.4)
plt.axvline(x=64, color='black', linestyle='--', alpha=0.4)
plt.axvline(x=best_params['beam_subdivisions'], color='red', linestyle='--', label=f"Best Beam Sub: {best_params['beam_subdivisions']}")
plt.legend()
plt.grid(True)
plt.xlabel("Beam Subdivisions")
plt.ylabel("RMSE")
plt.title("Parameter Sweep")
plt.subplot(1,2,2)
plt.plot(x, y, label='Data Points')
plt.axvline(x=best_params['beam_subdivisions'], color='red', linestyle='--', label=f"Best Beam Sub: {best_params['beam_subdivisions']}")
plt.legend()
plt.xlim(32, 64)
plt.ylim(0.09, 0.15)
plt.xlabel("Beam Subdivisions")
plt.ylabel("RMSE")
plt.title("Parameter Sweep (Zoomed)")
plt.grid(True)

plt.tight_layout()
plt.show()