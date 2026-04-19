import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import shutil
from skimage.metrics import structural_similarity as ssim

# =============================================================
# 1. FAN BEAM GEOMETRY SETUP
# =============================================================

def fan_setup(fan_angle, no_beams):
    """
    Returns a list of beam angles distributed symmetrically across
    the specified fan angle.
    """
    beam_angle_ls = []

    if no_beams <= 0:
        return beam_angle_ls

    if no_beams == 1:
        beam_angle_ls.append(0)

    elif no_beams % 2 != 0:
        # Odd: place a centre beam at 0 then symmetric pairs
        beam_angle_ls.append(0)
        for i in range(2, no_beams - 1, 2):
            angle = fan_angle * i / ((no_beams - 1) * 2)
            beam_angle_ls.insert(0, -angle)
            beam_angle_ls.append(angle)

    else:
        # Even: no centre beam, symmetric pairs then outer edges
        for i in range(2, no_beams - 1, 2):
            angle = fan_angle * i / (no_beams * 2)
            beam_angle_ls.insert(0, -angle)
            beam_angle_ls.append(angle)
        beam_angle_ls.insert(0, -fan_angle / 2)
        beam_angle_ls.append(fan_angle / 2)

    print(f"Fan setup complete: {len(beam_angle_ls)} beams over ±{np.degrees(fan_angle/2):.1f}°")
    return beam_angle_ls


# =============================================================
# 2. FORWARD PROJECTION  (builds A and b via Beer-Lambert Law)
# =============================================================

def ring_thing(fan_list, ring_subdivisions, beam_subdivisions,
               aperture, image_string, resize=64):
    """
    Constructs the system matrix A and measurement vector b by
    simulating a ring of fan-beam emitters around the image.

    Returns
    -------
    A    : ndarray (R x M)  – binary ray-pixel hit matrix
    b    : ndarray (R,)     – log-intensity measurements
    img  : ndarray (N x N x 3) – loaded/resized image
    """
    # --- Load and resize image ---
    image_path = os.path.join("test_images", image_string)
    img_raw = cv.imread(image_path)
    img_raw = cv.resize(img_raw, (resize, resize))
    img = np.flipud(cv.cvtColor(img_raw, cv.COLOR_BGR2RGB))
    shape = img.shape
    midpoint = np.flip(np.array([k / 2 for k in shape[:2]]))

    fan_angle = max(fan_list) - min(fan_list)

    # --- Matrix dimensions ---
    R = ring_subdivisions * len(fan_list)   # total rays
    M = shape[0] * shape[1]                  # total pixels
    A = np.zeros((R, M))
    b = np.zeros(R)

    # --- Ring radius: large enough to fully encircle the image ---
    if shape[0] <= shape[1]:
        ring_rad = (shape[1] * np.tan(np.pi / 2 - fan_angle)) / 2 + shape[0] / 2
    else:
        ring_rad = (shape[0] * np.tan(np.pi / 2 - fan_angle)) / 2 + shape[1] / 2

    print("Ring radius calculated. Building A matrix...")

    ray_index = 0

    for i in range(ring_subdivisions):
        angle = (2 * np.pi * i / ring_subdivisions) * aperture
        start_pos = (np.array([ring_rad * np.cos(angle),
                                ring_rad * np.sin(angle)]) + midpoint)

        # End positions for each beam in the fan
        end_pos_ls = [
            start_pos - np.array([2 * ring_rad * np.cos(angle - j),
                                   2 * ring_rad * np.sin(angle - j)])
            for j in fan_list
        ]

        for end_pos in end_pos_ls:
            xs = np.trunc(np.linspace(start_pos[0], end_pos[0], beam_subdivisions)).astype(int)
            ys = np.trunc(np.linspace(start_pos[1], end_pos[1], beam_subdivisions)).astype(int)

            pixel_values = []
            for dx, dy in zip(xs, ys):
                if 0 <= dx < shape[1] and 0 <= dy < shape[0]:
                    pixel_index = dy * shape[1] + dx
                    A[ray_index, pixel_index] = 1
                    pixel_values.append(img[dy, dx, 0])

            if len(pixel_values) > 0:
                pv = np.clip(np.array(pixel_values, dtype=float), 10, None)
                b[ray_index] = np.log(np.prod(pv / 255))
            else:
                b[ray_index] = 0

            ray_index += 1

        if (i + 1) % max(1, ring_subdivisions // 10) == 0:
            print(f"  Ring progress: {i+1}/{ring_subdivisions}")

    print(f"A matrix complete — shape: {A.shape}, b shape: {b.shape}")
    return A, b, img


# =============================================================
# 3. ITERATIVE SOLVER  (Kaczmarz / ART)
# =============================================================

def ART_solver(A, b, num_iterations=10):
    """
    Solves Ax = b iteratively using the Kaczmarz projection update:
        x^(k+1) = x^(k) + ((b_i - a_i · x^(k)) / ||a_i||²) * a_i
    """
    x = np.zeros(A.shape[1])

    for iteration in range(num_iterations):
        for i in range(len(b)):
            a_i = A[i]
            norm_sq = np.dot(a_i, a_i)
            if norm_sq == 0:
                continue
            x += ((b[i] - np.dot(a_i, x)) / norm_sq) * a_i
        print(f"  ART iteration {iteration + 1}/{num_iterations} complete")

    return x


# =============================================================
# 4. EVALUATION METRICS
# =============================================================

def get_edge_sharpness(image, global_min, global_max, threshold_ratio=0.15):
    """
    Measures mean Sobel gradient magnitude after Gaussian smoothing
    and background-ring suppression via thresholding.

    Returns
    -------
    mean_sharpness : float        – mean edge magnitude
    clean_magnitude: ndarray      – thresholded gradient map
    """
    scaled = (image - global_min) / (global_max - global_min + 1e-8)
    blurred = cv.GaussianBlur(scaled.astype(np.float32), (5, 5), 0)

    grad_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    threshold = np.max(magnitude) * threshold_ratio
    clean_magnitude = np.where(magnitude > threshold, magnitude, 0)

    return np.mean(clean_magnitude), clean_magnitude


def compute_metrics(true_img, recon_img, g_min, g_max):
    """
    Returns edge preservation (%) and SSIM (%) for a reconstruction.
    """
    s_true, _ = get_edge_sharpness(true_img, g_min, g_max)
    s_recon, edge_map = get_edge_sharpness(recon_img, g_min, g_max)
    preservation = (s_recon / s_true) * 100

    data_range = g_max - g_min
    ssim_score = ssim(true_img, recon_img, data_range=data_range) * 100

    return preservation, ssim_score, edge_map


# =============================================================
# 5. BASELINE RECONSTRUCTION
# =============================================================

def run_baseline(A, b, true_img, N, save_path="project/Images/diagnostic_reconstruction.png"):
    """
    Runs ART at default parameters, computes both quality metrics,
    and saves the three-panel diagnostic figure.
    """
    print("\nRunning baseline ART reconstruction (20 iterations)...")
    x_recon = ART_solver(A, b, num_iterations=20)
    recon_img = np.flipud(x_recon.reshape(N, N))

    g_min, g_max = np.min(true_img), np.max(true_img)
    preservation, ssim_score, edge_map = compute_metrics(true_img, recon_img, g_min, g_max)

    print(f"\n--- DIAGNOSTIC QUALITY REPORT ---")
    print(f"Edge Preservation : {preservation:.2f}%")
    print(f"SSIM              : {ssim_score:.2f}%")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(true_img, cmap='gray')
    axes[0].set_title("Ground Truth\n(Input Phantom)", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(recon_img, cmap='gray')
    axes[1].set_title(
        f"ART Reconstruction\n"
        f"SSIM: {ssim_score:.1f}%  |  Edge Preservation: {preservation:.1f}%",
        fontweight='bold'
    )
    axes[1].axis('off')

    axes[2].imshow(edge_map, cmap='magma')
    axes[2].set_title("Convolutional Edge Map\n(Sobel Gradient)", fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show(block=False)
    plt.pause(2)

    return recon_img, preservation, ssim_score


# =============================================================
# 6. PARAMETER ANALYSIS  (iterations sweep)
# =============================================================

def run_iteration_analysis(A, b, true_img, N,
                            save_path="project/Images/iteration_analysis.png"):
    """
    Sweeps over iteration counts and records edge preservation
    and SSIM at each point. Saves the dual-metric plot.
    """
    iteration_tests = [1, 2, 5, 10, 20, 30, 50]
    sharpness_scores = []
    ssim_scores = []

    g_min, g_max = np.min(true_img), np.max(true_img)
    s_true, _ = get_edge_sharpness(true_img, g_min, g_max)
    data_range = g_max - g_min

    print("\n--- Starting Iteration Parameter Analysis ---")

    for iters in iteration_tests:
        print(f"  Testing {iters} iteration(s)...")
        x_recon = ART_solver(A, b, num_iterations=iters)
        recon_img = np.flipud(x_recon.reshape(N, N))

        # Apply median denoising before scoring
        clean = cv.medianBlur(recon_img.astype(np.float32), 3)

        s_recon, _ = get_edge_sharpness(clean, g_min, g_max)
        sharpness_scores.append((s_recon / s_true) * 100)
        ssim_scores.append(ssim(true_img, clean, data_range=data_range) * 100)

    # --- Dual-axis plot ---
    fig, ax1 = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(bottom=0.2)

    color_sharp = 'tab:red'
    color_ssim  = 'tab:blue'

    ax1.set_xlabel('Number of ART Iterations', fontweight='bold')
    ax1.set_ylabel('Edge Sharpness Preservation (%)', color=color_sharp, fontweight='bold')
    line1 = ax1.plot(iteration_tests, sharpness_scores, marker='o', linestyle='-',
                     color=color_sharp, linewidth=2.5, label='Edge Sharpness')
    ax1.tick_params(axis='y', labelcolor=color_sharp)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Structural Similarity — SSIM (%)', color=color_ssim, fontweight='bold')
    line2 = ax2.plot(iteration_tests, ssim_scores, marker='s', linestyle='-',
                     color=color_ssim, linewidth=2.5, label='SSIM')
    ax2.tick_params(axis='y', labelcolor=color_ssim)

    plt.title("ART Iteration Analysis: Edge Sharpness vs. SSIM",
              fontsize=14, fontweight='bold')

    lines  = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=True, ncol=2, fontsize=10)

    fig.tight_layout(rect=[0, 0.15, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()

    return iteration_tests, sharpness_scores, ssim_scores


# =============================================================
# 7. MAIN PIPELINE
# =============================================================

if __name__ == "__main__":
    # --- Config ---
    IMAGE_FILE       = "test_image.jpg"
    N                = 64          # working resolution (N x N)
    FAN_ANGLE        = np.pi / 4
    NO_BEAMS         = 64
    RING_SUBDIVISIONS = 180
    BEAM_SUBDIVISIONS = 100

    # --- Setup folders ---
    os.makedirs("test_images", exist_ok=True)
    os.makedirs("project/Images", exist_ok=True)

    if not os.path.exists(f"test_images/{IMAGE_FILE}"):
        shutil.copy(IMAGE_FILE, f"test_images/{IMAGE_FILE}")

    # --- Build system ---
    fan_list = fan_setup(FAN_ANGLE, no_beams=NO_BEAMS)
    A, b, img = ring_thing(
        fan_list,
        ring_subdivisions=RING_SUBDIVISIONS,
        beam_subdivisions=BEAM_SUBDIVISIONS,
        aperture=1,
        image_string=IMAGE_FILE,
        resize=N
    )

    # Ground truth: grayscale, flipped to match reconstruction orientation
    true_img = np.flipud(cv.cvtColor(img, cv.COLOR_RGB2GRAY).astype(np.float64))

    # --- Baseline ---
    recon_img, base_preservation, base_ssim = run_baseline(
        A, b, true_img, N,
        save_path="project/Images/diagnostic_reconstruction.png"
    )

    # --- Parameter analysis ---
    run_iteration_analysis(
        A, b, true_img, N,
        save_path="project/Images/iteration_analysis.png"
    )