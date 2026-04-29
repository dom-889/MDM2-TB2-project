# Test range: from narrow 15° to very wide 120° (in radians)
angle_tests_deg = [15, 30, 45, 60, 90, 120]
angle_tests_rad = [np.radians(a) for a in angle_tests_deg]

sharpness_angle = []
ssim_angle = []

for angle in angle_tests_rad:
    print(f"Testing Fan Angle: {np.degrees(angle):.0f} degrees...")
    
    # 1. Setup geometry with fixed 96 beams and 90 ring subdivisions
    current_fan = fan_setup(angle, no_beams=96)
    
    # 2. Forward projection
    A_ang, b_ang, _ = ring_thing(current_fan, 
                                 ring_subdivisions=90, 
                                 beam_subdivisions=100, 
                                 aperture=1, 
                                 image_string="phantom.png", 
                                 resize=N)
    
    # 3. Reconstruction & Metrics (20 iterations for stability)
    x_rec = ART_solver(A_ang, b_ang, num_iterations=20)
    clean_rec = cv.medianBlur(x_rec.reshape(N,N).astype(np.float32), 3)
    
    s_rec, _ = get_edge_sharpness(clean_rec, g_min, g_max)
    sharpness_angle.append((s_rec / s_true) * 100)
    ssim_angle.append(ssim(true_img, clean_rec, data_range=data_range) * 100)

# ---------------------------------------------------------
# 8. PLOT FOR FAN ANGLE
# ---------------------------------------------------------
plt.rcParams.update({'font.size': 16})
fig, ax1 = plt.subplots(figsize=(7, 5))

color = 'tab:red'
ax1.set_xlabel('Fan Angle (Degrees)', fontweight='bold', fontsize=18)
ax1.set_ylabel('Sharpness (%)', color=color, fontweight='bold', fontsize=18)
ax1.plot(angle_tests_deg, sharpness_angle, marker='o', color=color, linewidth=3)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('SSIM (%)', color=color, fontweight='bold', fontsize=18)
ax2.plot(angle_tests_deg, ssim_angle, marker='s', color=color, linewidth=3)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Quality vs. Fan Aperture", fontsize=20, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig("project/Images/fan_angle_analysis.png", dpi=300)
plt.show()