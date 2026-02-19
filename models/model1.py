import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from  functions_file import get_runtime

# model 1 uses a ct scan type of method (yet to actually be implemented)
# but the high-level rundown is this: bro really forgot to give an explaination (its in the code so it should be fine)

def video_capture():

    # opens video feed and then can be used to take an image for testing using SPACE
    # might want to change the gray varaible when we get aroudn to actually processing more complex data

    cap = cv.VideoCapture(1)
    cv.namedWindow('test')

    if not cap.isOpened():

        print("Cant open camera")
        exit()

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Cant recieve frame (stream end?). Exiting ..")
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('test', gray)

        k = cv.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        elif k%256 == 32:
            # SPACE pressed
            img_name = "test_image.png"
            cv.imwrite(f"test_images/{img_name}", gray)
            print(f"{img_name} written!")
            cv.imshow(f"{img_name}", gray)
            
            
    cap.release()
    cv.destroyAllWindows()
    



@get_runtime
def grid_setup(fan_angle, no_beams, ring_subdivisions, beam_subdivisions, image_string, show_plot):
    beam_angle_ls = []
    image_path = f"test_images/{image_string}"
    img = np.flipud(np.array(cv.cvtColor(cv.imread(image_path), cv.COLOR_RGB2BGR))) # change the string here to the image you want to use

    fan_angle = np.pi/4
    shape = img.shape
    midpoint = np.flip(np.array([k/2 for k in shape[:2]]))
    colour_channels = 1
    
    def attenuation_calc(ls):
            ls = np.array(ls)
            if len(ls) == 0:
                return 1
            else:
                #return np.nanprod(ls/255)
                return np.log10(np.nanprod(ls/255))

    # general varable/list/dict setup yk how it be

    if no_beams > 0:
        if no_beams == 1:
            beam_angle_ls.append(0)
        elif no_beams - 2 >= 0:
            if no_beams % 2 != 0:
                beam_angle_ls.append(0)
            if no_beams % 2 != 0:
                for i in range(2,no_beams-1,2):
                    angle = fan_angle*(i)/((no_beams-1)*2)
                    beam_angle_ls.append(angle)
                    beam_angle_ls.insert(0,-angle)
            else:        
                for i in range(2,no_beams-1,2):
                    angle = fan_angle*(i)/((no_beams)*2)
                    beam_angle_ls.append(angle)
                    beam_angle_ls.insert(0,-angle)
            beam_angle_ls.insert(0,-fan_angle/2)
            beam_angle_ls.append(fan_angle/2)
    print("Fan angle list has been completed")
    # calculates angles for specified fan angle and number of beams and appends them to a list
    # highkey might end myself cause linspace exists although i will test speeds
    # this is actually faster when no of beams is < 2000 so it fits our usecase (i am better than numpy (massive lie))

    if shape[0] <= shape[1]:
        ring_rad = (shape[1]*np.tan((np.pi)/2-fan_angle))/2 + shape[0]/2
    elif shape[0] > shape[1]:
        ring_rad = (shape[0]*np.tan((np.pi)/2-fan_angle))/2 + shape[1]/2
    print("Ring radius has been calculated")
    # ring radius calculation
    
    fig, ax = plt.subplots(1,2,figsize=(9,4))
    beam_intensities = []
    for i in range(ring_subdivisions):

        
        # i think this works for calculating the mean values

        x_pos = []
        y_pos = []
        end_pos_ls = []
        angle = 2*np.pi*i/ring_subdivisions
        start_pos = np.array([ring_rad*np.cos(angle),ring_rad*np.sin(angle)]) + midpoint

        for j in beam_angle_ls:
            end_pos = start_pos - np.array([2*ring_rad*np.cos(angle-j), 2*ring_rad*np.sin(angle-j)])
            #print(f"{end_pos} = {start_pos} - {np.array([2*ring_rad*np.cos(angle-j), 2*ring_rad*np.sin(angle-j)])}")
            end_pos_ls.append(end_pos)
            # this bit calculates the end positions of each of the beams being fired in the 'fan'
        for colour in range(colour_channels):

            # i think this needs to be moved out to a different point or turn this next bit and stuff to functions cause this doesnt actually run properly and the value in the attenuation calc shouldnt be a fixed value

            for ii in end_pos_ls:
                atn_coef_ls = []
                x_subdivisions = np.trunc(np.linspace(start_pos[0], ii[0], beam_subdivisions))
                y_subdivisions = np.trunc(np.linspace(start_pos[1], ii[1], beam_subdivisions))
                # trunc used here cause all the pixel positions are integer values

                for dx, dy in zip(x_subdivisions, y_subdivisions):
                    if 0 <= dx < img.shape[1] and 0 <= dy < img.shape[0]:
                        atn_coef_ls.append(img[int(dy), int(dx), colour])
                    else:
                        atn_coef_ls.append(np.nan)
                # man i love it when i forget about a super useful function (zip)
                # use of nan here cause it retains index info and can be ignored by using some functions
                beam_intensities.append(attenuation_calc(atn_coef_ls))
            
            
                
            '''if show_plot:
                x_vals = []; y_vals = []
                for idx, intensity in enumerate(beam_intensities):
                    x_vals.append(idx)
                    y_vals.append(intensity)
                ax[1].set_ylim(0,1)
                ax[1].plot(x_vals, y_vals)'''
                

        if show_plot:
            '''for ii in end_pos_ls:
                x_pos.append(ii[0])
                y_pos.append(ii[1])
            ax[0].set_ylim((midpoint[1]-ring_rad), (midpoint[1]+ring_rad))
            ax[0].set_xlim((midpoint[0]-ring_rad), (midpoint[0]+ring_rad))
            ax[0].plot(x_pos, y_pos)
            ax[0].plot(midpoint[0], midpoint[1])
            ax[0].plot(start_pos[0], start_pos[1])'''
            ax[0].imshow(np.flipud(img))
            ax[0].axis('off')
            ax[0].set_title('Input image')
        print(f"Beam vector calculations in progress [{i+1} of {ring_subdivisions} complete]")
    print("Beam vector calculations complete")
    beam_intensities = np.reshape(beam_intensities,(ring_subdivisions, no_beams))
    ax[1].imshow(beam_intensities, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title(f"Sinogram produced from image")
    if show_plot:
        plt.show()