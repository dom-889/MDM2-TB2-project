import numpy as np
import matplotlib.pyplot as plt
from fixed_model import fan_setup, ring_thing, ART_solver


def noise_analysis(A, b, noise_levels, num_iterations=10, num_trials=5):
    """
    For each noise level sigma, adds Gaussian noise to b num_trials times,
    runs ART, and records the mean RMSE against the clean reconstruction.
    """
    print("Running clean reconstruction...")
    x_clean = ART_solver(A, b, num_iterations=num_iterations)

    mean_errors = []
    std_errors = []

    for sigma in noise_levels:
        print(f"  Noise level sigma = {sigma:.4f}")
        trial_errors = []

        for trial in range(num_trials):
            noise = np.random.normal(0, sigma, size=b.shape)
            b_noisy = b + noise
            x_noisy = ART_solver(A, b_noisy, num_iterations=num_iterations)
            rmse = np.sqrt(np.mean((x_noisy - x_clean) ** 2))
            trial_errors.append(rmse)

        mean_errors.append(np.mean(trial_errors))
        std_errors.append(np.std(trial_errors))
        print(f"    Mean RMSE = {np.mean(trial_errors):.5f} ± {np.std(trial_errors):.5f}")

    return np.array(mean_errors), np.array(std_errors), x_clean


if __name__ == "__main__":

    phantoms = ["bone_phantom.png", "lung_phantom.png", "anomaly_phantom.png"]
    labels = ["Bone", "Lung", "Anomaly"]
    noise_levels = np.linspace(0, 0.5, 11)
    resize = 64

    # store results for each phantom
    all_results = {}
    all_Ab = {}

    for name, label in zip(phantoms, labels):
        print(f"\n{'='*50}")
        print(f"Processing: {label} phantom")
        print(f"{'='*50}")

        fan_list = fan_setup(np.pi/4, no_beams=64)
        A, b, img = ring_thing(fan_list, ring_subdivisions=180,
                               beam_subdivisions=100, aperture=1,
                               image_string=name, resize=resize)

        mean_errors, std_errors, x_clean = noise_analysis(
            A, b,
            noise_levels=noise_levels,
            num_iterations=10,
            num_trials=5
        )

        all_results[label] = {
            "mean": mean_errors,
            "std": std_errors,
            "clean": x_clean,
            "img": img
        }
        all_Ab[label] = (A, b)

    # --- PLOT 1: RMSE vs noise for all three phantoms on one graph ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = ["steelblue", "indianred", "seagreen"]

    for (label, data), colour in zip(all_results.items(), colours):
        ax.errorbar(noise_levels, data["mean"], yerr=data["std"],
                    fmt='o-', capsize=4, color=colour, label=label)

    ax.set_xlabel('Noise level (σ)', fontsize=12)
    ax.set_ylabel('RMSE vs clean reconstruction', fontsize=12)
    ax.set_title('ART Reconstruction Error vs Gaussian Noise Level', fontsize=13)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/results/noise_analysis_all_phantoms.png', dpi=150)
    print("Saved noise_analysis_all_phantoms.png")

    # --- PLOT 2: original vs clean reconstruction for each phantom ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    for idx, (label, data) in enumerate(all_results.items()):
        # original image
        axes[0, idx].imshow(np.flipud(data["img"][:, :, 0]), cmap='gray')
        axes[0, idx].set_title(f'{label} — Original')
        axes[0, idx].axis('off')

        # clean ART reconstruction
        axes[1, idx].imshow(np.flipud(data["clean"].reshape(resize, resize)), cmap='gray')
        axes[1, idx].set_title(f'{label} — ART Reconstruction')
        axes[1, idx].axis('off')

    plt.suptitle('Original Images vs Clean ART Reconstructions', fontsize=14)
    plt.tight_layout()
    plt.savefig('images/results/phantom_reconstructions.png', dpi=150)
    print("Saved phantom_reconstructions.png")

    # --- PLOT 3: noisy reconstructions side by side for each phantom ---
    selected_sigmas = [0.05, 0.1, 0.25, 0.5]
    n_cols = len(selected_sigmas) + 1
    fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 9))

    for row, (label, data) in enumerate(all_results.items()):
        A, b = all_Ab[label]

        # clean
        axes[row, 0].imshow(np.flipud(data["clean"].reshape(resize, resize)), cmap='gray')
        axes[row, 0].set_title(f'{label}\nClean (σ=0)')
        axes[row, 0].axis('off')

        # noisy
        for col, sigma in enumerate(selected_sigmas):
            noise = np.random.normal(0, sigma, size=b.shape)
            b_noisy = b + noise
            x_noisy = ART_solver(A, b_noisy, num_iterations=10)
            axes[row, col + 1].imshow(np.flipud(x_noisy.reshape(resize, resize)), cmap='gray')
            axes[row, col + 1].set_title(f'σ = {sigma}')
            axes[row, col + 1].axis('off')

    plt.suptitle('ART Reconstructions at Increasing Noise Levels', fontsize=14)
    plt.tight_layout()
    plt.savefig('images/results/noise_reconstruction_comparison.png', dpi=150)
    print("Saved noise_reconstruction_comparison.png")

    plt.show()