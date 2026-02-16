import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def grid_setup(fan_angle, no_beams, ring_subdivisions):
    beam_angle_ls = []
    ring_dict = {}

    img = np.array(cv.imread("test_images/test_image.png"))
    fan_angle = np.pi/4
    shape = img.shape
    midpoint = np.array([k/2 for k in shape[:2]])
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

    for i in range(ring_subdivisions):
        end_pos_ls = []
        angle = 2*np.pi*i/ring_subdivisions
        start_pos = np.array([ring_rad*np.cos(angle),ring_rad*np.sin(angle)]) + midpoint
        for j in beam_angle_ls:
            end_pos = start_pos - np.array([2*ring_rad*np.cos(j-angle), 2*ring_rad*np.sin(j-angle)])
            end_pos_ls.append(end_pos)
        print(end_pos_ls)
        plt.plot(end_pos_ls)
        plt.show()
        break
        
grid_setup(np.pi/4, 10, 360)






