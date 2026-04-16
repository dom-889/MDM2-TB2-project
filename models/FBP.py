import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from fixed_model_new import fan_setup, ring_thing, ART_solver
from scipy import ndimage

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

        #Apply ramp filter in frequency domain
        r_fft = np.fft.fft(r,axis=0)

        if i < window_iterations:
            r_fft_filtered = r_fft * ramp_window
            r_filtered = np.real(np.fft.ifft(r_fft_filtered))
        else:
            r_fft_filtered = r_fft * ramp
            r_filtered = np.real(np.fft.ifft(r_fft_filtered))

        #Iterative step
        delta_x = A.T @ r_filtered #Think this is the backprojection?
        delta_x /= col_norm #Normalize by column sums

        x += lambda_val * delta_x #Update the image estimate

        print(f"FBP iteration {i+1} of {num_iterations} complete")

    return x


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

def compute_rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

n = 64

image_name = "shepp_logan_phantom.png"
phantom = cv.imread(f"test_images/{image_name}", cv.IMREAD_GRAYSCALE)
#phantom = cv.resize(phantom, (n, n), interpolation=cv.INTER_AREA)
phantom = phantom.astype(np.float32) / 255.0
ref = cv.resize(phantom, (n, n), interpolation=cv.INTER_AREA).flatten()
ref_norm = (ref - np.min(ref)) / (np.max(ref) - np.min(ref))


sigma = 0.05
noisy_image = phantom + np.random.normal(0, sigma, phantom.shape)
noisy_image = np.clip(noisy_image, 0, 1)
cv.imwrite("test_images/temp_noisy.png", (noisy_image * 255).astype(np.uint8))

w_iterations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
#w_iterations = [0, 5, 10, 15, 20]

w_strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#w_strengths = [0.2, 0.4, 0.6, 0.8, 1.0]

rmses = np.zeros((len(w_iterations), len(w_strengths)))
edge_rmses = np.zeros((len(w_iterations), len(w_strengths)))

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
x_ref_norm = (x_ref_corrected - np.min(x_ref_corrected)) / (np.max(x_ref_corrected) - np.min(x_ref_corrected))

fan_list = fan_setup(np.pi/4, no_beams=128)
A, b, img = ring_thing(fan_list,
                           ring_subdivisions=360,
                           beam_subdivisions=100,
                           aperture=1,
                           image_string="temp_noisy.png",
                           resize=n)

best_rmse = None
best_edge_rmse = None
best_params = None
best_params_edge = None

# Parameter sweep for window strength and iterations
for i, w_iter in enumerate(w_iterations):
    for j, w_str in enumerate(w_strengths):
        x_fbp = FBP_window_solver(A, b, num_iterations=25, lambda_val=1, window_strength=w_str, window_iterations=w_iter)
        x_fbp_image = np.flipud(x_fbp.reshape(n, n))
        x_norm = (x_fbp_image - np.min(x_fbp_image)) / (np.max(x_fbp_image) - np.min(x_fbp_image))
        #x_norm = x_norm ** 1.5
        rmse = compute_rmse(x_ref_norm.flatten(), x_norm.flatten())
        edg_rmse = edge_rmse(x_ref_norm.reshape(n, n), x_norm.reshape(n, n))
        rmses[i, j] = rmse
        edge_rmses[i, j] = edg_rmse
        print(f"Window Iterations: {w_iter}, Window Strength: {w_str}, RMSE: {rmse:.4f}")
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_params = (w_iter, w_str)
        if best_edge_rmse is None or edg_rmse < best_edge_rmse:
            best_edge_rmse = edg_rmse
            best_params_edge = (w_iter, w_str)

print(f"Best RMSE: {best_rmse:.4f} with parameters: {best_params}")
print(f"Best Edge RMSE: {best_edge_rmse:.4f} with parameters: {best_params_edge}")

optimal_w_iter, optimal_w_str = best_params
x_fbp_optimal = FBP_window_solver(A, b, num_iterations=25, lambda_val=0.8, window_strength=optimal_w_str, window_iterations=optimal_w_iter)
x_fbp_image_optimal = np.flipud(x_fbp_optimal.reshape(n, n))
x_norm_optimal = (x_fbp_image_optimal - np.min(x_fbp_image_optimal)) / (np.max(x_fbp_image_optimal) - np.min(x_fbp_image_optimal))
x_norm_optimal = x_norm_optimal ** 1.5

optimal_w_iter_edge, optimal_w_str_edge = best_params_edge
x_fbp_optimal_edge = FBP_window_solver(A, b, num_iterations=25, lambda_val=0.8, window_strength=optimal_w_str_edge, window_iterations=optimal_w_iter_edge)
x_fbp_image_optimal_edge = np.flipud(x_fbp_optimal_edge.reshape(n, n))
x_norm_optimal_edge = (x_fbp_image_optimal_edge - np.min(x_fbp_image_optimal_edge)) / (np.max(x_fbp_image_optimal_edge) - np.min(x_fbp_image_optimal_edge))
x_norm_optimal_edge = x_norm_optimal_edge ** 1.5



plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(noisy_image, cmap='gray')
plt.title('Original  Noisy Phantom')
plt.text(0.5, -0.1, f'Noise Std Dev: {sigma}', ha='center', va='center', transform=plt.gca().transAxes)
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(x_norm_optimal, cmap='gray')
plt.title('FBP Reconstruction')
#plt.xlabel(f'RMSE: {best_rmse:.4f}, Window Iterations: {optimal_w_iter}, Window Strength: {optimal_w_str}')
plt.text(0.5, -0.1, f'RMSE: {best_rmse:.4f}\n Window Iterations: {optimal_w_iter}\n Window Strength: {optimal_w_str}', 
         ha='center', va='center', transform=plt.gca().transAxes)
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(x_norm_optimal_edge, cmap='gray')
plt.title('FBP Reconstruction (Edge-Optimized)')
plt.text(0.5, -0.1, f'Edge RMSE: {best_edge_rmse:.4f}\n Window Iterations: {optimal_w_iter_edge}\n Window Strength: {optimal_w_str_edge}', 
         ha='center', va='center', transform=plt.gca().transAxes)
plt.axis('off')

plt.figure()
plt.subplot(1,2,1)
plt.imshow(rmses, cmap='hot', extent=[min(w_strengths), max(w_strengths), max(w_iterations), min(w_iterations)], aspect='auto')
plt.title('RMSE Heatmap')
plt.xlabel('Window Strength')
plt.ylabel('Window Iterations')
plt.colorbar(label='RMSE')

plt.subplot(1,2,2)
plt.imshow(edge_rmses, cmap='hot', extent=[min(w_strengths), max(w_strengths), max(w_iterations), min(w_iterations)], aspect='auto')
plt.title('Edge RMSE Heatmap')
plt.xlabel('Window Strength')
plt.ylabel('Window Iterations')
plt.colorbar(label='Edge RMSE')

plt.tight_layout()

plt.show()
print(f'Optimal parameters: Window Iterations = {optimal_w_iter}, Window Strength = {optimal_w_str}, RMSE = {best_rmse:.4f}')
