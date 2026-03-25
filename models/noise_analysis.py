import numpy as np
import matplotlib.pyplot as plt
from main import fan_setup, ring_thing, ART_solver


def ART_solver(A, b, num_iterations=10):
    """
    Solves Ax = b iteratively using the Kaczmarz (ART) update:
        x = x + ((b_i - a_i . x) / |a_i|^2) * a_i
    """
    M = A.shape[1]
    x = np.zeros(M)
    for iteration in range(num_iterations):
        for i in range(len(b)):
            a_i = A[i]
            norm_sq = np.dot(a_i, a_i)
            if norm_sq == 0:
                continue
            residual = b[i] - np.dot(a_i, x)
            x = x + (residual / norm_sq) * a_i
    return x


def noise_analysis(A, b, img, noise_levels, num_iterations=10, num_trials=5, resize=64):
    """
    For each noise level sigma, adds Gaussian noise to b num_trials times,
    runs ART, and records the mean RMSE against the clean reconstruction.

    Parameters
    ----------
    A              : ray-pixel path matrix (R x M)
    b              : clean log-intensity measurement vector (R,)
    img            : original image array (used as ground truth)
    noise_levels   : list/array of sigma values to test
    num_iterations : ART iterations per solve
    num_trials     : number of noisy trials per sigma (for averaging)
    resize         : image side length (must match A/b construction)

    Returns
    -------
    mean_errors    : mean RMSE for each noise level
    std_errors     : std of RMSE across trials for each noise level
    x_clean        : clean ART reconstruction (for plotting)
    """

    # ground truth: flatten and normalise the original image to [0,1]
    ground_truth = img[:, :, 0].flatten().astype(float) / 255.0

    # clean reconstruction (no noise) — used as reference
    print("Running clean reconstruction...")
    x_clean = ART_solver(A, b, num_iterations=num_iterations)

    mean_errors = []
    std_errors = []

    for sigma in noise_levels:
        print(f"\nNoise level sigma = {sigma:.4f}")
        trial_errors = []

        for trial in range(num_trials):
            # add Gaussian noise to the measurement vector
            noise = np.random.normal(0, sigma, size=b.shape)
            b_noisy = b + noise

            # reconstruct from noisy measurements
            x_noisy = ART_solver(A, b_noisy, num_iterations=num_iterations)

            # RMSE against ground truth
            rmse = np.sqrt(np.mean((x_noisy - ground_truth) ** 2))
            trial_errors.append(rmse)

        mean_errors.append(np.mean(trial_errors))
        std_errors.append(np.std(trial_errors))
        print(f"  Mean RMSE = {np.mean(trial_errors):.5f} ± {np.std(trial_errors):.5f}")

    return np.array(mean_errors), np.array(std_errors), x_clean


def plot_noise_analysis(noise_levels, mean_errors, std_errors):
    """
    Plots RMSE vs noise level (sigma) with error bars.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(noise_levels, mean_errors, yerr=std_errors,
                fmt='o-', capsize=4, color='steelblue',
                label='Mean RMSE ± std')

    ax.set_xlabel('Noise level (σ)', fontsize=12)
    ax.set_ylabel('RMSE vs ground truth', fontsize=12)
    ax.set_title('ART Reconstruction Error vs Gaussian Noise Level', fontsize=13)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('noise_analysis_plot.png', dpi=150)
    plt.show()
    print("Plot saved to noise_analysis_plot.png")


def plot_reconstructions_at_noise_levels(A, b, img, selected_sigmas,
                                         num_iterations=10, resize=64):
    """
    Shows side-by-side reconstructions at a few chosen noise levels,
    so you can visually see degradation.
    """
    n = len(selected_sigmas) + 1  # +1 for the clean reconstruction
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))

    # clean
    x_clean = ART_solver(A, b, num_iterations=num_iterations)
    axes[0].imshow(np.flipud(x_clean.reshape(resize, resize)), cmap='gray')
    axes[0].set_title('Clean (σ=0)')
    axes[0].axis('off')

    for idx, sigma in enumerate(selected_sigmas):
        noise = np.random.normal(0, sigma, size=b.shape)
        b_noisy = b + noise
        x_noisy = ART_solver(A, b_noisy, num_iterations=num_iterations)
        axes[idx + 1].imshow(np.flipud(x_noisy.reshape(resize, resize)), cmap='gray')
        axes[idx + 1].set_title(f'σ = {sigma}')
        axes[idx + 1].axis('off')

    plt.suptitle('ART Reconstructions at Increasing Noise Levels', fontsize=12)
    plt.tight_layout()
    # plt.savefig('noise_reconstruction_comparison.png', dpi=150)
    plt.show()
    print("Comparison plot saved to noise_reconstruction_comparison.png")

if __name__ == "__main__":

    fan_list = fan_setup(np.pi/4, no_beams=64)
    A, b, img = ring_thing(fan_list, ring_subdivisions=180,
                           beam_subdivisions=100, aperture=1,
                           image_string="big_john.jpg", resize=64)
    
    # Noise analysis
    noise_levels = np.linspace(0, 0.5, 11)

    mean_errors, std_errors, x_clean = noise_analysis(
        A, b, img,
        noise_levels=noise_levels,
        num_iterations=10,
        num_trials=5,
        resize=64
    )

    # Error plot
    plot_noise_analysis(noise_levels, mean_errors, std_errors)

    # Visual comparison of reconstructions at different noise levels
    plot_reconstructions_at_noise_levels(
        A, b, img,
        selected_sigmas=[0.05, 0.1, 0.25, 0.5],
        num_iterations=10,
        resize=64
    )