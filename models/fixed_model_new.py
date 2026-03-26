import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


def fan_setup(fan_angle, no_beams):

    beam_angle_ls = []
    fan_angle = np.pi/4

    if no_beams > 0:
        if no_beams == 1:
            beam_angle_ls.append(0)
        elif no_beams - 2 >= 0:
            if no_beams % 2 != 0:
                beam_angle_ls.append(0)
            if no_beams % 2 != 0:
                for i in range(2, no_beams-1, 2):
                    angle = fan_angle*(i)/((no_beams-1)*2)
                    beam_angle_ls.append(angle)
                    beam_angle_ls.insert(0, -angle)
            else:
                for i in range(2, no_beams-1, 2):
                    angle = fan_angle*(i)/((no_beams)*2)
                    beam_angle_ls.append(angle)
                    beam_angle_ls.insert(0, -angle)
            beam_angle_ls.insert(0, -fan_angle/2)
            beam_angle_ls.append(fan_angle/2)

    print("Fan angle list has been completed")
    return beam_angle_ls


def ring_thing(fan_list, ring_subdivisions, beam_subdivisions, aperture, image_string, resize=64):

    # load and resize image 
    image_path = f"test_images/{image_string}"
    img_raw = cv.imread(image_path)
    img_raw = cv.resize(img_raw, (resize, resize))  # resize to keep A manageable
    img = np.flipud(np.array(cv.cvtColor(img_raw, cv.COLOR_RGB2BGR)))
    shape = img.shape
    midpoint = np.flip(np.array([k/2 for k in shape[:2]]))
    fan_angle = max(fan_list) - min(fan_list)

    # matrix dimensions 
    R = ring_subdivisions * len(fan_list)   # total number of rays
    M = shape[0] * shape[1]                 # total number of pixels
    A = np.zeros((R, M))                    # path matrix: R rows x M cols
    b = np.zeros(R)                         # measurement vector: one value per ray

    # ring radius 
    if shape[0] <= shape[1]:
        ring_rad = (shape[1] * np.tan((np.pi)/2 - fan_angle))/2 + shape[0]/2
    else:
        ring_rad = (shape[0] * np.tan((np.pi)/2 - fan_angle))/2 + shape[1]/2
    print("Ring radius has been calculated")

    ray_index = 0  # tracks which row of A and b we are filling

    for i in range(ring_subdivisions):

        angle = (2 * np.pi * i / ring_subdivisions) * aperture
        start_pos = np.array([ring_rad * np.cos(angle),
                               ring_rad * np.sin(angle)]) + midpoint

        # calculate end positions for each beam in the fan
        end_pos_ls = []
        for j in fan_list:
            end_pos = start_pos - np.array([2 * ring_rad * np.cos(angle - j),
                                            2 * ring_rad * np.sin(angle - j)])
            end_pos_ls.append(end_pos)

        for ii in end_pos_ls:

            # sample pixel positions along the ray
            x_subdivisions = np.trunc(np.linspace(start_pos[0], ii[0], beam_subdivisions))
            y_subdivisions = np.trunc(np.linspace(start_pos[1], ii[1], beam_subdivisions))

            pixel_values = []

            for dx, dy in zip(x_subdivisions, y_subdivisions):
                if 0 <= dx < shape[1] and 0 <= dy < shape[0]:

                    # flatten 2D pixel index to 1D for the matrix column
                    pixel_index = int(dy) * shape[1] + int(dx)

                    # mark this pixel as hit by this ray in A
                    A[ray_index, pixel_index] = 1

                    # collect pixel value for b
                    pixel_values.append(img[int(dy), int(dx), 0])

            # compute log-intensity measurement for this ray
            if len(pixel_values) > 0:
                pixel_values = np.array(pixel_values, dtype=float)
                pixel_values = np.clip(pixel_values, 10, None)  # avoid log(0)
                b[ray_index] = np.log10(np.prod(pixel_values / 255))
            else:
                b[ray_index] = 0

            ray_index += 1

        print(f"Ring progress: [{i+1} of {ring_subdivisions} complete]")

    print("A matrix and b vector construction complete")
    print(f"A shape: {A.shape}, b shape: {b.shape}")

    return A, b, img


def ART_solver(A, b, num_iterations=10):
    """
    Solves Ax = b iteratively using the Kaczmarz (ART) update:
        x = x + ((b_i - a_i . x) / |a_i|^2) * a_i
    """
    M = A.shape[1] # number of pixels
    x = np.zeros(M)  # initial guess: all zeros

    # perform ART iterations
    # each iteration goes through all rays and updates the solution x
    for iteration in range(num_iterations):
        for i in range(len(b)):
            a_i = A[i]
            norm_sq = np.dot(a_i, a_i)
            if norm_sq == 0:
                continue  # skip empty rays
            residual = b[i] - np.dot(a_i, x)
            x = x + (residual / norm_sq) * a_i

        print(f"ART iteration {iteration+1} of {num_iterations} complete")

    return x


if __name__ == "__main__":

    # make sure test_images folder exists 
    os.makedirs("test_images", exist_ok=True)
    """
    # copy big_john.jpg into test_images if not already there 
    # make sure big_john.jpg is in the same folder as this script
    if not os.path.exists("test_images/hand_xray.jpg"):
        import shutil
        shutil.copy("hand_xray.jpg", "test_images/hand_xray.jpg")
    """
    # run the pipeline 
    fan_list = fan_setup(np.pi/4, no_beams=64)
    A, b, img = ring_thing(fan_list,
                           ring_subdivisions=180,
                           beam_subdivisions=100,
                           aperture=1,
                           image_string="hand_xray.jpg",
                           resize=64)  # resize controls image resolution, increase for better quality but slower

    x = ART_solver(A, b, num_iterations=20)

    # reshape reconstruction back to 2D 
    reconstruction = x.reshape(64, 64)
    # reshape and flip to correct the spatial inversion 
    reconstruction = np.flipud(reconstruction)

    # display results 
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    ax[0].imshow(cv.cvtColor(cv.resize(cv.imread("test_images/hand_xray.jpg"), (64, 64)), cv.COLOR_BGR2RGB))
    ax[0].set_title("Original image")
    ax[0].axis('off')

    ax[1].spy(A, markersize=0.1)
    ax[1].set_title("A matrix (ray-pixel hits)")
    ax[1].axis('off')

    ax[2].imshow(reconstruction, cmap='gray')
    ax[2].set_title("ART reconstruction")
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()


