import numpy as np
import matplotlib.pyplot as plt
from fixed_model import fan_setup, ring_thing, ART_solver


def run_reconstruction(image_string, no_beams, ring_subdivisions, 
                       beam_subdivisions=100, resize=64, num_iterations=20,
                       fan_angle=np.pi/4):
    """
    Runs the full pipeline for given parameters and returns the reconstruction,
    plus A, b, and img for reuse.
    """
    fan_list = fan_setup(fan_angle, no_beams=no_beams)
    A, b, img = ring_thing(fan_list, ring_subdivisions=ring_subdivisions,
                           beam_subdivisions=beam_subdivisions, aperture=1,
                           image_string=image_string, resize=resize)
    x = ART_solver(A, b, num_iterations=num_iterations)
    return x, A, b, img


# =====================================================================
# DEFAULT SETTINGS (used as baseline when not being varied)
# =====================================================================
IMAGE = "bone_phantom.png"
RESIZE = 64
DEFAULT_RING = 180
DEFAULT_BEAMS = 64
DEFAULT_BEAM_SUBS = 100
DEFAULT_ITERATIONS = 20
DEFAULT_FAN_ANGLE = np.pi / 4


if __name__ == "__main__":

    # =================================================================
    # TEST 1: VARY RING SUBDIVISIONS
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Varying ring subdivisions")
    print("=" * 60)

    ring_values = [15, 30, 60, 90, 180, 360]

    fig, axes = plt.subplots(1, len(ring_values) + 1, 
                             figsize=(3 * (len(ring_values) + 1), 3.5))

    # show original
    _, _, _, img = run_reconstruction(IMAGE, DEFAULT_BEAMS, ring_values[0], 
                                      resize=RESIZE, num_iterations=1)
    axes[0].imshow(np.flipud(img[:, :, 0]), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    ring_reconstructions = {}
    for idx, ring_sub in enumerate(ring_values):
        print(f"\n  Ring subdivisions = {ring_sub}")
        x, _, _, _ = run_reconstruction(IMAGE, DEFAULT_BEAMS, ring_sub, resize=RESIZE)
        ring_reconstructions[ring_sub] = x
        axes[idx + 1].imshow(np.flipud(x.reshape(RESIZE, RESIZE)), cmap='gray')
        axes[idx + 1].set_title(f'Ring = {ring_sub}')
        axes[idx + 1].axis('off')

    plt.suptitle(f'Effect of Ring Subdivisions (beams = {DEFAULT_BEAMS})', fontsize=13)
    plt.tight_layout()
    plt.savefig('test1_ring_subdivisions.png', dpi=150)
    print("Saved test1_ring_subdivisions.png")

    # =================================================================
    # TEST 2: VARY NUMBER OF BEAMS
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Varying number of beams")
    print("=" * 60)

    beam_values = [8, 16, 32, 64, 128, 256]

    fig, axes = plt.subplots(1, len(beam_values) + 1, 
                             figsize=(3 * (len(beam_values) + 1), 3.5))

    axes[0].imshow(np.flipud(img[:, :, 0]), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    beam_reconstructions = {}
    for idx, beams in enumerate(beam_values):
        print(f"\n  Beams = {beams}")
        x, _, _, _ = run_reconstruction(IMAGE, beams, DEFAULT_RING, resize=RESIZE)
        beam_reconstructions[beams] = x
        axes[idx + 1].imshow(np.flipud(x.reshape(RESIZE, RESIZE)), cmap='gray')
        axes[idx + 1].set_title(f'Beams = {beams}')
        axes[idx + 1].axis('off')

    plt.suptitle(f'Effect of Number of Beams (ring subdivisions = {DEFAULT_RING})', fontsize=13)
    plt.tight_layout()
    plt.savefig('test2_beams.png', dpi=150)
    print("Saved test2_beams.png")

    # =================================================================
    # TEST 3: RMSE VS TOTAL RAYS (ring subs and beams on same graph)
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST 3: RMSE vs total rays")
    print("=" * 60)

    # reference: highest resolution run
    print("  Generating reference reconstruction...")
    x_ref, _, _, _ = run_reconstruction(IMAGE, 256, 360, resize=RESIZE)

    ring_rmse = []
    ring_total_rays = []
    for ring_sub in ring_values:
        x = ring_reconstructions[ring_sub]
        rmse = np.sqrt(np.mean((x - x_ref) ** 2))
        ring_rmse.append(rmse)
        ring_total_rays.append(ring_sub * DEFAULT_BEAMS)
        print(f"  Ring={ring_sub}, total rays={ring_sub * DEFAULT_BEAMS}, RMSE={rmse:.5f}")

    beam_rmse = []
    beam_total_rays = []
    for beams in beam_values:
        x = beam_reconstructions[beams]
        rmse = np.sqrt(np.mean((x - x_ref) ** 2))
        beam_rmse.append(rmse)
        beam_total_rays.append(DEFAULT_RING * beams)
        print(f"  Beams={beams}, total rays={DEFAULT_RING * beams}, RMSE={rmse:.5f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ring_total_rays, ring_rmse, 'o-', color='steelblue', label='Varying ring subdivisions')
    ax.plot(beam_total_rays, beam_rmse, 's-', color='indianred', label='Varying beams per fan')
    ax.set_xlabel('Total number of rays', fontsize=12)
    ax.set_ylabel('RMSE vs high-res reference', fontsize=12)
    ax.set_title('Reconstruction Quality vs Number of Rays', fontsize=13)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('test3_rmse_vs_rays.png', dpi=150)
    print("Saved test3_rmse_vs_rays.png")

    # =================================================================
    # TEST 4: VARY ART ITERATIONS (semi-convergence analysis)
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST 4: Varying ART iterations")
    print("=" * 60)

    iteration_values = [1, 2, 5, 10, 20, 50, 100]

    # build A and b once, then solve with different iteration counts
    fan_list = fan_setup(DEFAULT_FAN_ANGLE, no_beams=DEFAULT_BEAMS)
    A, b, img = ring_thing(fan_list, ring_subdivisions=DEFAULT_RING,
                           beam_subdivisions=DEFAULT_BEAM_SUBS, aperture=1,
                           image_string=IMAGE, resize=RESIZE)

    fig, axes = plt.subplots(1, len(iteration_values) + 1, 
                             figsize=(3 * (len(iteration_values) + 1), 3.5))

    axes[0].imshow(np.flipud(img[:, :, 0]), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    iter_reconstructions = {}
    for idx, n_iter in enumerate(iteration_values):
        print(f"\n  Iterations = {n_iter}")
        x = ART_solver(A, b, num_iterations=n_iter)
        iter_reconstructions[n_iter] = x
        axes[idx + 1].imshow(np.flipud(x.reshape(RESIZE, RESIZE)), cmap='gray')
        axes[idx + 1].set_title(f'Iter = {n_iter}')
        axes[idx + 1].axis('off')

    plt.suptitle(f'Effect of ART Iterations (ring={DEFAULT_RING}, beams={DEFAULT_BEAMS})', fontsize=13)
    plt.tight_layout()
    plt.savefig('test4_iterations.png', dpi=150)
    print("Saved test4_iterations.png")

    # RMSE vs iterations (using highest iteration as reference)
    x_ref_iter = iter_reconstructions[max(iteration_values)]
    # also compare against ground truth image
    ground_truth = img[:, :, 0].flatten().astype(float) / 255.0

    iter_rmse_vs_ref = []
    iter_rmse_vs_gt = []
    for n_iter in iteration_values:
        x = iter_reconstructions[n_iter]
        rmse_ref = np.sqrt(np.mean((x - x_ref_iter) ** 2))
        rmse_gt = np.sqrt(np.mean((x - ground_truth) ** 2))
        iter_rmse_vs_ref.append(rmse_ref)
        iter_rmse_vs_gt.append(rmse_gt)
        print(f"  Iter={n_iter}, RMSE vs ref={rmse_ref:.5f}, RMSE vs ground truth={rmse_gt:.5f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iteration_values, iter_rmse_vs_gt, 'o-', color='steelblue', label='RMSE vs ground truth')
    ax.plot(iteration_values, iter_rmse_vs_ref, 's-', color='indianred', label='RMSE vs 100-iter reference')
    ax.set_xlabel('Number of ART iterations', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('ART Convergence: Reconstruction Error vs Iterations', fontsize=13)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('test4_iterations_rmse.png', dpi=150)
    print("Saved test4_iterations_rmse.png")

    # =================================================================
    # TEST 5: VARY BEAM SUBDIVISIONS (ray sampling density)
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST 5: Varying beam subdivisions")
    print("=" * 60)

    beam_sub_values = [10, 25, 50, 100, 200, 400]

    fig, axes = plt.subplots(1, len(beam_sub_values) + 1, 
                             figsize=(3 * (len(beam_sub_values) + 1), 3.5))

    axes[0].imshow(np.flipud(img[:, :, 0]), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    beamsub_reconstructions = {}
    for idx, bsub in enumerate(beam_sub_values):
        print(f"\n  Beam subdivisions = {bsub}")
        x, _, _, _ = run_reconstruction(IMAGE, DEFAULT_BEAMS, DEFAULT_RING,
                                        beam_subdivisions=bsub, resize=RESIZE)
        beamsub_reconstructions[bsub] = x
        axes[idx + 1].imshow(np.flipud(x.reshape(RESIZE, RESIZE)), cmap='gray')
        axes[idx + 1].set_title(f'Sub = {bsub}')
        axes[idx + 1].axis('off')

    plt.suptitle(f'Effect of Beam Subdivisions (ring={DEFAULT_RING}, beams={DEFAULT_BEAMS})', fontsize=13)
    plt.tight_layout()
    plt.savefig('test5_beam_subdivisions.png', dpi=150)
    print("Saved test5_beam_subdivisions.png")

    # RMSE for beam subdivisions
    x_ref_bsub = beamsub_reconstructions[max(beam_sub_values)]
    beamsub_rmse = []
    for bsub in beam_sub_values:
        rmse = np.sqrt(np.mean((beamsub_reconstructions[bsub] - x_ref_bsub) ** 2))
        beamsub_rmse.append(rmse)
        print(f"  BeamSub={bsub}, RMSE={rmse:.5f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(beam_sub_values, beamsub_rmse, 'o-', color='steelblue')
    ax.set_xlabel('Beam subdivisions (samples per ray)', fontsize=12)
    ax.set_ylabel('RMSE vs high-res reference', fontsize=12)
    ax.set_title('Reconstruction Quality vs Ray Sampling Density', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('test5_beam_subdivisions_rmse.png', dpi=150)
    print("Saved test5_beam_subdivisions_rmse.png")

    # =================================================================
    # TEST 6: VARY FAN ANGLE
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST 6: Varying fan angle")
    print("=" * 60)

    fan_angles = [np.pi/8, np.pi/6, np.pi/4, np.pi/3, np.pi/2.5, np.pi/2]
    fan_labels = ['π/8', 'π/6', 'π/4', 'π/3', '2π/5', 'π/2']

    fig, axes = plt.subplots(1, len(fan_angles) + 1, 
                             figsize=(3 * (len(fan_angles) + 1), 3.5))

    axes[0].imshow(np.flipud(img[:, :, 0]), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    fan_reconstructions = {}
    for idx, (fa, fl) in enumerate(zip(fan_angles, fan_labels)):
        print(f"\n  Fan angle = {fl}")
        x, _, _, _ = run_reconstruction(IMAGE, DEFAULT_BEAMS, DEFAULT_RING,
                                        fan_angle=fa, resize=RESIZE)
        fan_reconstructions[fl] = x
        axes[idx + 1].imshow(np.flipud(x.reshape(RESIZE, RESIZE)), cmap='gray')
        axes[idx + 1].set_title(f'Fan = {fl}')
        axes[idx + 1].axis('off')

    plt.suptitle(f'Effect of Fan Angle (ring={DEFAULT_RING}, beams={DEFAULT_BEAMS})', fontsize=13)
    plt.tight_layout()
    plt.savefig('test6_fan_angle.png', dpi=150)
    print("Saved test6_fan_angle.png")

    # RMSE for fan angles (use pi/4 as reference since it's the default)
    x_ref_fan = fan_reconstructions['π/4']
    fan_rmse = []
    fan_degrees = []
    for fa, fl in zip(fan_angles, fan_labels):
        rmse = np.sqrt(np.mean((fan_reconstructions[fl] - x_ref_fan) ** 2))
        fan_rmse.append(rmse)
        fan_degrees.append(np.degrees(fa))
        print(f"  Fan={fl} ({np.degrees(fa):.1f}°), RMSE vs π/4 ref={rmse:.5f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fan_degrees, fan_rmse, 'o-', color='steelblue')
    ax.set_xlabel('Fan angle (degrees)', fontsize=12)
    ax.set_ylabel('RMSE vs π/4 reference', fontsize=12)
    ax.set_title('Reconstruction Quality vs Fan Angle', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('test6_fan_angle_rmse.png', dpi=150)
    print("Saved test6_fan_angle_rmse.png")

    # =================================================================
    # TEST 7: VARY IMAGE RESOLUTION
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST 7: Varying image resolution")
    print("=" * 60)

    resize_values = [16, 32, 64, 128]

    fig, axes = plt.subplots(2, len(resize_values), 
                             figsize=(4 * len(resize_values), 7))

    for idx, res in enumerate(resize_values):
        print(f"\n  Resolution = {res}x{res}")
        x, _, _, res_img = run_reconstruction(IMAGE, DEFAULT_BEAMS, DEFAULT_RING,
                                              resize=res)
        # original
        axes[0, idx].imshow(np.flipud(res_img[:, :, 0]), cmap='gray')
        axes[0, idx].set_title(f'Original {res}x{res}')
        axes[0, idx].axis('off')

        # reconstruction
        axes[1, idx].imshow(np.flipud(x.reshape(res, res)), cmap='gray')
        axes[1, idx].set_title(f'ART {res}x{res}')
        axes[1, idx].axis('off')

    plt.suptitle(f'Effect of Image Resolution (ring={DEFAULT_RING}, beams={DEFAULT_BEAMS})', fontsize=13)
    plt.tight_layout()
    plt.savefig('test7_resolution.png', dpi=150)
    print("Saved test7_resolution.png")

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
    print("Saved files:")
    print("  test1_ring_subdivisions.png")
    print("  test2_beams.png")
    print("  test3_rmse_vs_rays.png")
    print("  test4_iterations.png")
    print("  test4_iterations_rmse.png")
    print("  test5_beam_subdivisions.png")
    print("  test5_beam_subdivisions_rmse.png")
    print("  test6_fan_angle.png")
    print("  test6_fan_angle_rmse.png")
    print("  test7_resolution.png")

    plt.show()