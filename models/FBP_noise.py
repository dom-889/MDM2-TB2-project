import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from fixed_model_new import fan_setup, ring_thing, ART_solver

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

def compute_rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

n = 64

image_name = "shepp_logan_phantom.png"
phantom = cv.imread(f"test_images/{image_name}", cv.IMREAD_GRAYSCALE)
#phantom = cv.resize(phantom, (n, n), interpolation=cv.INTER_AREA)
phantom = phantom.astype(np.float32) / 255.0
ref = cv.resize(phantom, (n, n), interpolation=cv.INTER_AREA).flatten()
ref_norm = (ref - np.min(ref)) / (np.max(ref) - np.min(ref))

w_iterations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
w_strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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

sigmas = [0, 0.05, 0.1, 0.15]

fig_imgs, axes_imgs = plt.subplots(2, len(sigmas), figsize=(3*len(sigmas), 7))
fig_heatmaps, axes_heatmaps = plt.subplots(1, len(sigmas), figsize=(3*len(sigmas), 4))

for i, sigma in enumerate(sigmas):
    noisy_image = phantom + np.random.normal(0, sigma, phantom.shape)
    noisy_image = np.clip(noisy_image, 0, 1)
    cv.imwrite(f"test_images/temp_noisy_{int(sigma*100)}.png", (noisy_image * 255).astype(np.uint8))


    rmses = np.zeros((len(w_iterations), len(w_strengths)))


    fan_list = fan_setup(np.pi/4, no_beams=128)
    A, b, img = ring_thing(fan_list,
                           ring_subdivisions=360,
                           beam_subdivisions=100,
                           aperture=1,
                           image_string=f"temp_noisy_{int(sigma*100)}.png",
                           resize=n)

    best_rmse = None
    best_params = None

    # Parameter sweep for window strength and iterations
    for j, w_iter in enumerate(w_iterations):
        for k, w_str in enumerate(w_strengths):
            x_fbp = FBP_window_solver(A, b, num_iterations=25, lambda_val=0.8, window_strength=w_str, window_iterations=w_iter)
            x_fbp_image = np.flipud(x_fbp.reshape(n, n))
            x_norm = (x_fbp_image - np.min(x_fbp_image)) / (np.max(x_fbp_image) - np.min(x_fbp_image))
            #x_norm = x_norm ** 1.5
            rmse = compute_rmse(ref_norm.flatten(), x_norm.flatten())
            rmses[j, k] = rmse
            print(f"Window Iterations: {w_iter}, Window Strength: {w_str}, RMSE: {rmse:.4f}")
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_params = (w_iter, w_str)

    print(f"Best RMSE: {best_rmse:.4f} with parameters: {best_params}")

    optimal_w_iter, optimal_w_str = best_params
    x_fbp_optimal = FBP_window_solver(A, b, num_iterations=25, lambda_val=0.8, window_strength=optimal_w_str, window_iterations=optimal_w_iter)
    x_fbp_image_optimal = np.flipud(x_fbp_optimal.reshape(n, n))
    x_norm_optimal = (x_fbp_image_optimal - np.min(x_fbp_image_optimal)) / (np.max(x_fbp_image_optimal) - np.min(x_fbp_image_optimal))
    x_norm_optimal = x_norm_optimal ** 1.5


    axes_imgs[0,i].imshow(noisy_image, cmap='gray')
    axes_imgs[0,i].set_title('Original Phantom')
    axes_imgs[0,i].text(0.5, -0.06, f'Noise σ: {sigma}', ha='center', va='center', transform=axes_imgs[0,i].transAxes)
    axes_imgs[0,i].axis('off')

    axes_imgs[1,i].imshow(x_norm_optimal, cmap='gray')
    axes_imgs[1,i].set_title('FBP Reconstruction')
    #plt.xlabel(f'RMSE: {best_rmse:.4f}, Window Iterations: {optimal_w_iter}, Window Strength: {optimal_w_str}')
    axes_imgs[1,i].text(0.5, -0.15, f'RMSE: {best_rmse:.4f}\n Window Iterations: {optimal_w_iter}\n Window Strength: {optimal_w_str}', 
         ha='center', va='center', transform=axes_imgs[1,i].transAxes)
    axes_imgs[1,i].axis('off')

    ax = axes_heatmaps[i]
    im = ax.imshow(rmses, cmap='hot', 
                   extent=[min(w_strengths), max(w_strengths), max(w_iterations), min(w_iterations)],
                   aspect='auto',
                   )
    ax.set_title(f"σ={sigma}")
    ax.set_xlabel("Window Strength")
    ax.set_ylabel("Window Iterations")
    fig_heatmaps.colorbar(im, ax=ax)

    '''
    axes_heatmaps[i].imshow(rmses, cmap='hot', extent=[min(w_strengths), max(w_strengths), max(w_iterations), min(w_iterations)], aspect='auto')
    axes_heatmaps[i].set_title('RMSE Heatmap')
    axes_heatmaps[i].set_xlabel('Window Strength')
    axes_heatmaps[i].set_ylabel('Window Iterations')
    axes_heatmaps[i].colorbar(label='RMSE')
    '''
    
plt.tight_layout()
fig_imgs.suptitle("Noisy Phantoms and FBP Reconstructions", fontsize=16)
fig_heatmaps.suptitle("RMSE Heatmaps", fontsize=16)

plt.figure()
plt.imshow(x_ref_norm.reshape(n, n), cmap='gray')
plt.show()
#print(f'Optimal parameters: Window Iterations = {optimal_w_iter}, Window Strength = {optimal_w_str}, RMSE = {best_rmse:.4f}')
