import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from fixed_model_new import fan_setup, ring_thing, ART_solver
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
import skimage.filters
from skimage.morphology import square

def FBP_window_solver(A, b, num_iterations=10, lambda_val=1.0, window_strength=0.5, window_iterations=5):
    """
    Solves Ax = b using iterative Filtered Back Projection (FBP):
        Applies a ramp filter at each iteration to refine the image estimate.
        x^(k+1) = x^(k) +  lambda_val * (A^T * inverse_fourier{filter * fourier{b - A * x^*(k)}}) / column_norms

    """
    M, P = A.shape # M: rays, P: pixels
    x = np.zeros(P)  # initial guess of all zeros

    col_norm = np.sum(A, axis=0)
    col_norm[col_norm == 0] = 1 #Avoid division by zero

    ramp = np.abs(np.fft.fftfreq(M)) #Ramp filter in the frequency domain
    hann_window = 1 - window_strength + window_strength * np.hanning(M) #Hanning window to reduce ringing artifacts
    ramp_window = hann_window * ramp #Apply window to ramp filter

    for i in range(num_iterations):
        r = b - A @ x #Compute resdidual

        r_fft = np.fft.fft(r,axis=0) #Apply ramp filter in frequency domain

        if i < window_iterations: #Applies for the first i iterations, then switches to unwindowed ramp filter
            r_fft_filtered = r_fft * ramp_window
            r_filtered = np.real(np.fft.ifft(r_fft_filtered))
        else:
            r_fft_filtered = r_fft * ramp
            r_filtered = np.real(np.fft.ifft(r_fft_filtered))

        #Iterative step
        delta_x = A.T @ r_filtered 
        delta_x /= col_norm #Normalize by column sums

        x += lambda_val * delta_x #Update the image estimate

        print(f"FBP iteration {i+1} of {num_iterations} complete")

    return x

n = 64 #Size of image

image_name = "shepp_logan_phantom.png"
phantom = cv.imread(f"test_images/{image_name}", cv.IMREAD_GRAYSCALE)

phantom = phantom.astype(np.float32)/255
x_true = np.log10(np.clip(
    cv.cvtColor(cv.resize(cv.imread(f"test_images/{image_name}"), (n, n)), 
                cv.COLOR_BGR2GRAY).astype(float) / 255, 1e-6, None)).flatten()
true_img = x_true.reshape(n, n)
true_img = cv.resize(cv.imread(f"test_images/{image_name}", cv.IMREAD_GRAYSCALE), (n, n)).astype(np.float32)/255

sigma = 0 #STD dev of noise, Noiseless = 0
noisy_image = phantom + np.random.normal(0, sigma, phantom.shape)
noisy_image = np.clip(noisy_image, 0, 1)
cv.imwrite("test_images/temp_noisy.png", (noisy_image * 255).astype(np.uint8))

#ART high param reference imnage
fan_list_ref = fan_setup(np.pi/4, 128)
A_ref, b_ref, _ = ring_thing(fan_list_ref,
                            ring_subdivisions=360,
                            beam_subdivisions=61,
                            aperture=1,
                            image_string=image_name,
                            resize=n)
x_ref = ART_solver(A_ref, b_ref, num_iterations=35)
x_ref_corrected = np.flipud(x_ref.reshape(n, n)).flatten()
x_ref_norm = (x_ref_corrected - np.min(x_ref_corrected)) / (np.max(x_ref_corrected) - np.min(x_ref_corrected)) #Normalise from 0 to 1
true_img = x_ref_norm.reshape(n, n)

g_min = np.min(true_img)
g_max = np.max(true_img)
data_range = g_max - g_min


def get_edge_sharpness(image, global_min, global_max, threshold_ratio=0.12):
    """Measures gradient magnitude with noise suppression and thresholding."""
    scaled_image = (image - global_min) / (global_max - global_min + 1e-8)
    blurred = cv.GaussianBlur(scaled_image.astype(np.float32), (5, 5), 0)
    
    grad_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    max_edge = np.max(magnitude)
    clean_magnitude = np.where(magnitude > (max_edge * threshold_ratio), magnitude, 0)
    
    return np.mean(clean_magnitude), clean_magnitude

s_true, clean_edges = get_edge_sharpness(true_img, g_min, g_max, threshold_ratio=0.0)

iterations = 25

w_iterations = np.linspace(0, iterations, iterations+1, dtype=int) #Test across [0, max iterations]

w_strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

ssims = np.zeros((len(w_iterations), len(w_strengths)))          #Create empty arrays to store each result
preservations = np.zeros((len(w_iterations), len(w_strengths)))



fan_list = fan_setup(np.pi/4, no_beams=128)
A, b, img = ring_thing(fan_list,
                           ring_subdivisions=360,
                           beam_subdivisions=100,
                           aperture=1,
                           image_string="temp_noisy.png",
                           resize=n)

best_ssim = None
best_ssim_params = None
best_preservation = None
best_preservation_params = None

# Parameter sweep for window strength and iterations
for i, w_iter in enumerate(w_iterations):
    for j, w_str in enumerate(w_strengths):
        x_recon = FBP_window_solver(A, b, num_iterations=iterations, lambda_val=1, window_strength=w_str, window_iterations=w_iter)
        x_corrected = np.flipud(x_recon.reshape(n, n)) #FLip upside down to correct for ring application handles data
        x_norm = (x_corrected - np.min(x_corrected)) / (np.max(x_corrected) - np.min(x_corrected))
        temp_recon_image = x_norm.reshape(n, n)

        clean_recon = cv.medianBlur(temp_recon_image.astype(np.float32), 3)
        s_recon, edge_map = get_edge_sharpness(clean_recon, g_min, g_max)  #Change threshold here
        preservation = (s_recon / s_true) * 100 #Convert to percent
        preservations[i, j] = preservation
        current_ssim = ssim(true_img, clean_recon, data_range=data_range)
        current_ssim *= 100 #Convert to percent
        ssims[i, j] = current_ssim

        if best_ssim is None or current_ssim > best_ssim:  #Store the best configuration for both metrics
            best_ssim = current_ssim
            best_ssim_params = (w_iter, w_str)
            ssim_edge_map = edge_map
            ssim_recon = s_recon
        
        if best_preservation is None or preservation > best_preservation:
            best_preservation = preservation
            best_preservation_params = (w_iter, w_str)
            preservation_edge_map = edge_map

print(f'Best SSIM: {best_ssim:.4f}% with parameters: Window Iterations = {best_ssim_params[0]}, Window Strength = {best_ssim_params[1]}')
print(f'Best Preservation: {best_preservation:.2f}% with parameters: Window Iterations = {best_preservation_params[0]}, Window Strength = {best_preservation_params[1]}')

x_ssim_optimal = FBP_window_solver(A, b, num_iterations=iterations, lambda_val=1, window_strength=best_ssim_params[1], window_iterations=best_ssim_params[0])
x_ssim_optimal = np.flipud(x_ssim_optimal.reshape(n, n))

x_preservation_optimal = FBP_window_solver(A, b, num_iterations=iterations, lambda_val=1, window_strength=best_preservation_params[1], window_iterations=best_preservation_params[0])
x_preservation_optimal = np.flipud(x_preservation_optimal.reshape(n, n))

print(f'Optimal parameters: Window Iterations = {best_ssim_params[0]}, Window Strength = {best_ssim_params[1]}, SSIM = {best_ssim:.4f}')
print(g_min, g_max)
print(s_true)
print(ssim_recon)


plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.imshow(true_img, cmap='gray')
plt.title('Original Reference Phantom')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(x_ssim_optimal, cmap='gray')
plt.title(f'Optimal Reconstruction\nWindow Iterations {best_ssim_params[0]} | Strength: {best_ssim_params[1]}\nSSIM: {best_ssim:.2f}%  |  Edge Preservation: {best_preservation:.2f}%')
plt.axis('off')

plt.tight_layout()
plt.savefig('Shepp_Logan_results_final/FBP_images.png', dpi=300)


plt.figure(figsize=(9, 3))
plt.subplot(1,2,1)
plt.imshow(ssims.T, cmap='hot', extent=[min(w_iterations), max(w_iterations), min(w_strengths), max(w_strengths)], aspect='auto')
plt.title('SSIM Heatmap', fontsize=12, fontweight='bold')
plt.xlabel('Window Iterations', fontsize = 12)
plt.ylabel('Window Strength', fontsize = 12)
plt.colorbar(label='SSIM')

plt.subplot(1,2,2)
plt.imshow(preservations.T, cmap='hot', extent=[min(w_iterations), max(w_iterations), min(w_strengths), max(w_strengths)], aspect='auto')
plt.title('Edge Preservation Heatmap', fontsize=12, fontweight='bold')
plt.xlabel('Window Iterations', fontsize=12)
plt.ylabel('Window Strength', fontsize=12)
plt.colorbar(label='Edge Preservation (%)')

plt.tight_layout()
plt.savefig('Shepp_Logan_results_final/FBP_param_sweeps.png', dpi=300)


plt.figure(figsize =(9,4))
plt.subplot(1,2,1)
plt.imshow(clean_edges, cmap='hot')
plt.title("True Edge Map")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(ssim_edge_map, cmap='hot', vmax=np.max(clean_edges), vmin=0)
plt.title("Optimal Reconstruction Edge Map")
plt.axis('off')

plt.tight_layout()
plt.savefig('Shepp_Logan_results_final/FBP_edge_heat_maps.png', dpi=300)

plt.show()
print(f'Optimal parameters: Window Iterations = {best_ssim_params[0]}, Window Strength = {best_ssim_params[1]}, SSIM = {best_ssim:.4f}')
print(f'g_min: {g_min}, g_max: {g_max}, s_true: {s_true}, s_recon: {ssim_recon}')
