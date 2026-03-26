import numpy as np
import matplotlib.pyplot as plt
from fixed_model import fan_setup, ring_thing, ART_solver


IMAGE = "bone_phantom.png"
RESIZE = 64
DEFAULT_RING = 180
DEFAULT_BEAMS = 64
DEFAULT_ITERATIONS = 20


if __name__ == "__main__":

    fan_angles = [np.pi/8, np.pi/6, np.pi/4, np.pi/3, np.pi/2.5, np.pi/2]
    fan_labels = ['π/8', 'π/6', 'π/4', 'π/3', '2π/5', 'π/2']

    # --- VISUAL COMPARISON ---
    fig, axes = plt.subplots(1, len(fan_angles) + 1,
                             figsize=(3 * (len(fan_angles) + 1), 3.5))

    # get original image
    fan_list = fan_setup(np.pi/4, no_beams=DEFAULT_BEAMS)
    _, _, img = ring_thing(fan_list, ring_subdivisions=DEFAULT_RING,
                           beam_subdivisions=100, aperture=1,
                           image_string=IMAGE, resize=RESIZE)

    axes[0].imshow(np.flipud(img[:, :, 0]), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    reconstructions = {}
    for idx, (fa, fl) in enumerate(zip(fan_angles, fan_labels)):
        print(f"\nFan angle = {fl} ({np.degrees(fa):.1f}°)")
        fan_list = fan_setup(fa, no_beams=DEFAULT_BEAMS)
        A, b, _ = ring_thing(fan_list, ring_subdivisions=DEFAULT_RING,
                              beam_subdivisions=100, aperture=1,
                              image_string=IMAGE, resize=RESIZE)
        x = ART_solver(A, b, num_iterations=DEFAULT_ITERATIONS)
        reconstructions[fl] = x
        axes[idx + 1].imshow(np.flipud(x.reshape(RESIZE, RESIZE)), cmap='gray')
        axes[idx + 1].set_title(f'Fan = {fl}\n({np.degrees(fa):.0f}°)')
        axes[idx + 1].axis('off')

    plt.suptitle(f'Effect of Fan Angle (ring={DEFAULT_RING}, beams={DEFAULT_BEAMS})', fontsize=13)
    plt.tight_layout()
    plt.savefig('fan_angle_visual.png', dpi=150)
    print("\nSaved fan_angle_visual.png")

    # --- RMSE GRAPH ---
    # use ground truth image for comparison
    ground_truth = img[:, :, 0].flatten().astype(float) / 255.0

    # also compare each against the π/4 reconstruction as a relative reference
    x_ref = reconstructions['π/4']

    fan_degrees = []
    rmse_vs_ref = []
    rmse_vs_gt = []

    for fa, fl in zip(fan_angles, fan_labels):
        x = reconstructions[fl]
        r_ref = np.sqrt(np.mean((x - x_ref) ** 2))
        r_gt = np.sqrt(np.mean((x - ground_truth) ** 2))
        rmse_vs_ref.append(r_ref)
        rmse_vs_gt.append(r_gt)
        fan_degrees.append(np.degrees(fa))
        print(f"Fan={fl} ({np.degrees(fa):.1f}°), RMSE vs π/4={r_ref:.5f}, RMSE vs GT={r_gt:.5f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fan_degrees, rmse_vs_gt, 'o-', color='steelblue', label='RMSE vs ground truth')
    ax.plot(fan_degrees, rmse_vs_ref, 's-', color='indianred', label='RMSE vs π/4 reference')
    ax.set_xlabel('Fan angle (degrees)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Reconstruction Quality vs Fan Angle', fontsize=13)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('fan_angle_rmse.png', dpi=150)
    print("Saved fan_angle_rmse.png")

    plt.show() 