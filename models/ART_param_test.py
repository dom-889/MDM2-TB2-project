from fixed_model_new import fan_setup, ring_thing, ART_solver
#import os
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

image_name = "shepp_logan_phantom.png"

n = 64

phantom = cv.imread(f"test_images/{image_name}", cv.IMREAD_GRAYSCALE)
phantom = cv.resize(phantom, (n, n))

x_true = np.log10(np.clip(cv.resize(cv.imread(f"test_images/{image_name}", cv.IMREAD_GRAYSCALE),(n, n)).astype(float) / 255, 1e-6, None)).flatten()
true_img = x_true.reshape(n, n)



# compute RMSE 
def compute_rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))


ring_subdivisions = [360]
beam_sizes = [128]
beam_subdivisions = [8, 11, 16, 23, 32, 45, 64, 91, 128]
fan_angles = [np.pi/4]
iterations = [30]

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
                    rmse = compute_rmse(x_corrected, x_true)
                    rmse = rmse / (np.max(x_true) - np.min(x_true))  # Normalise RMSE

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

print(beam_sub_rmses)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(phantom, cmap='gray')
plt.title("Original Phantom")
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(x_best.reshape(n, n), cmap='gray')
plt.title(f"Reconstructed Image (RMSE: {best_rmse:.4f}, Beam Sub: {best_params['beam_subdivisions']})")
plt.axis('off')
plt.subplot(1,3,3)
plt.plot(beam_subdivisions, beam_sub_rmses, marker='o')
plt.grid(True)
plt.xlabel("Beam Subdivisions")
plt.ylabel("Normalised RMSE")
plt.title("Parameter Sweep")

plt.tight_layout()
plt.show()
print("\nBest Parameters:")
print(best_params)

'''
fan_list = fan_setup(np.pi/5, 360)
A, b, _ = ring_thing(fan_list,
                    ring_subdivisions=90,
                    beam_subdivisions=90,
                    aperture=1,
                    image_string=image_name,
                    resize=n)
x = ART_solver(A, b, num_iterations=50)
x_corrected = np.flipud(x.reshape(n, n)).flatten()
x_corrected = x_corrected.astype(float)
#x_ref_small = cv.resize(phantom, (n, n)).flatten().astype(float)
#rmse = compute_rmse(x_corrected, x_ref_small)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(phantom, cmap='gray')
plt.title("Original Phantom")
plt.subplot(1,2,2)
plt.imshow(x_corrected.reshape(n, n), cmap='gray')
plt.title(f"Reconstructed Image (RMSE: {rmse:.4f})")
plt.axis('off')
plt.tight_layout()
plt.show()
'''