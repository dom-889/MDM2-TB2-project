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
def grid_setup(fan_angle, no_beams, ring_subdivisions, show_plot, beam_subdivisions):
    beam_angle_ls = []
    ring_dict = {}

    img = np.flipud(np.array(cv.imread("test_images/test_image.png")))
    fan_angle = np.pi/4
    shape = img.shape
    midpoint = np.flip(np.array([k/2 for k in shape[:2]]))
    colour_channels = 1
    

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

    if shape[0] <= shape[1]:
        ring_rad = (shape[1]*np.tan((np.pi)/2-fan_angle))/2 + shape[0]/2
    elif shape[0] > shape[1]:
        ring_rad = (shape[0]*np.tan((np.pi)/2-fan_angle))/2 + shape[1]/2
    print("Ring radius has been calculated")
    # ring radius calculation
    
    fix, ax = plt.subplots(1,2,figsize=(9,4))

    for i in range(ring_subdivisions):

        def attenuation_calc(ls):
            ls = np.array(ls)
            print(ls)
            if len(ls) == 0:
                return 1
            else:
                return np.mean(ls)/255


        x_pos = []
        y_pos = []
        end_pos_ls = []
        angle = 2*np.pi*i/ring_subdivisions
        start_pos = np.array([ring_rad*np.cos(angle),ring_rad*np.sin(angle)]) + midpoint

        for j in beam_angle_ls:
            end_pos = start_pos - np.array([2*ring_rad*np.cos(angle-j), 2*ring_rad*np.sin(angle-j)])
            #print(f"{end_pos} = {start_pos} - {np.array([2*ring_rad*np.cos(angle-j), 2*ring_rad*np.sin(angle-j)])}")
            end_pos_ls.append(end_pos)

        for colour in range(colour_channels):
            beam_intensities = []
            for ii in end_pos_ls:
                atn_coef_ls = []
                x_subdivisions = np.trunc(np.linspace(start_pos[0], ii[0], beam_subdivisions))
                y_subdivisions = np.trunc(np.linspace(start_pos[1], ii[1], beam_subdivisions))
                if shape[0] > shape[1]:

                    for index, dx in enumerate(x_subdivisions):
                        if 0 <= dx < shape[1]:
                            if 0 <= y_subdivisions[index] < shape[0]:
                                atn_coef = img[int(y_subdivisions[index])][int(dx)]
                                atn_coef_ls.append(atn_coef[colour_channels])
                else:
                    for index, dy in enumerate(y_subdivisions):
                        if 0 <= dy < shape[0]:
                            if 0 <= x_subdivisions[index] < shape[0]:
                                atn_coef = img[int(x_subdivisions[index])][int(dy)]
                                atn_coef_ls.append(atn_coef[colour_channels])

                beam_intensities.append(attenuation_calc(atn_coef_ls))

            if show_plot:
                x_vals = []; y_vals = []
                for idx, intensity in enumerate(beam_intensities):
                    x_vals.append(idx)
                    y_vals.append(intensity)
                ax[1].plot(x_vals, y_vals)

                    



            
            




            
            



        if show_plot:
            for ii in end_pos_ls:
                x_pos.append(ii[0])
                y_pos.append(ii[1])
            ax[0].set_ylim((midpoint[1]-2*ring_rad), (midpoint[1]+2*ring_rad))
            ax[0].set_xlim((midpoint[0]-2*ring_rad), (midpoint[0]+2*ring_rad))
            ax[0].plot(x_pos, y_pos)
            ax[0].plot(midpoint[0], midpoint[1])
            ax[0].plot(start_pos[0], start_pos[1])
            ax[0].imshow(img)
        print(f"Beam vector calculations in progress [{i+1} of {ring_subdivisions} complete]")
    print("Beam vector calculations complete")
    if show_plot:
        plt.show()