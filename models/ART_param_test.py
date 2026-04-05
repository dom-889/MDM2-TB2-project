from fixed_model_new import fan_setup, ring_thing, ART_solver
#import os
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

image_name = "shepp_logan_phantom.png"

'''
ring_subdivisions = [30, 60, 90, 180, 360]
beam_sizes = [8, 16, 32, 64, 128]
beam_subdivisions = [10, 50, 100, 200]
fan_angles = [np.pi/8, np.pi/6, np.pi/4, np.pi/3,np.pi/2]
iterations = [5, 10, 20, 50]

phantom = cv.imread(f"test_images/{image_name}", cv.IMREAD_GRAYSCALE)
x_ref = phantom.flatten()
resize_ref = phantom.shape[0] 

def compute_rmse(x, x_ref):
    return np.sqrt(np.mean((x - x_ref)**2))

best_rmse = float('inf')
best_params = None

print("Starting parameter sweep...")
for ring_sub in ring_subdivisions:
    for beam_size in beam_sizes:
         for fan_angle in fan_angles:
            fan_list = fan_setup(fan_angle, no_beams=beam_size)
            for beam_sub in beam_subdivisions:
                resize_test = 64
                A, b, _ = ring_thing(fan_list,
                                        ring_subdivisions=ring_sub,
                                        beam_subdivisions=beam_sub,
                                        aperture=1,
                                        image_string=image_name,
                                        resize=resize_test)
                for num_iter in iterations:
                    x = ART_solver(A, b, num_iterations=num_iter)
                    x_ref_small = cv.resize(phantom, (resize_test, resize_test)).flatten()
                    rmse = compute_rmse(x, x_ref_small)
                    print(f"Ring: {ring_sub}, Beam: {beam_size}, Beam Sub: {beam_sub}, Iter: {num_iter} -> RMSE: {rmse:.4f}")

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = {
                            "ring_subdivisions": ring_sub,
                            "beam_sizes": beam_size,
                            "beam_subdivisions": beam_sub,
                            "iterations": num_iter
                        }

print("\nBest Parameters:")
print(best_params)
print(f"Best RMSE: {best_rmse:.4f}")

fan_list_best = fan_setup(np.pi/4, no_beams=best_params["beam_sizes"])
A_best, b_best, _ = ring_thing(fan_list_best,
                            ring_subdivisions=best_params["ring_subdivisions"],
                            beam_subdivisions=best_params["beam_subdivisions"],
                            aperture=1,
                            image_string=image_name,
                            resize=64)
x_best = ART_solver(A_best, b_best, num_iterations=best_params["iterations"])
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(phantom, cmap='gray')
plt.title("Original Phantom")
plt.subplot(1,2,2)
plt.imshow(x_best.reshape(64, 64), cmap='gray')
plt.title("Reconstructed Image")
plt.axis('off')
plt.tight_layout()
plt.show()
'''

def compute_rmse(x, x_ref):
    return np.sqrt(np.mean((x - x_ref)**2))

n = 100
phantom = cv.imread(f"test_images/{image_name}", cv.IMREAD_GRAYSCALE)

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

phantom_float = phantom.astype(float) / 255.0
x_corrected_float = x_corrected / np.max(x_corrected)  # normalize reconstruction
rmse = compute_rmse(x_corrected_float, cv.resize(phantom_float, (n,n)).flatten())

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