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
def grid_setup(fan_angle, fan_subdivisions, ring_subdivisions):
    fan_sd_ls = []

    img = np.array(cv.imread("test_images\\test_image.png"))
    fan_angle = np.pi/4
    shape = img.shape
    if shape[0] <= shape[1]:
        ring_rad = (shape[1]*np.tan((np.pi)/2-fan_angle))/2 + shape[0]/2
    elif shape[0] > shape[1]:
        ring_rad = (shape[0]*np.tan((np.pi)/2-fan_angle))/2 + shape[1]/2
    print("Ring radius has been calculated")

    if fan_subdivisions > 0:
        if fan_subdivisions == 1:
            fan_sd_ls.append(0)
        elif fan_subdivisions - 2 >= 0:
            if fan_subdivisions % 2 != 0:
                fan_sd_ls.append(0)
            if fan_subdivisions % 2 != 0:
                for i in range(2,fan_subdivisions-1,2):
                    angle = fan_angle*(i)/((fan_subdivisions-1)*2)
                    fan_sd_ls.append(angle)
                    fan_sd_ls.insert(0,-angle)
            else:        
                for i in range(2,fan_subdivisions-1,2):
                    angle = fan_angle*(i)/((fan_subdivisions)*2)
                    fan_sd_ls.append(angle)
                    fan_sd_ls.insert(0,-angle)
            fan_sd_ls.insert(0,-fan_angle/2)
            fan_sd_ls.append(fan_angle/2)
    print("Fan angle list has been completed")
    for i in range(ring_subdivisions):
        