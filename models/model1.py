import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def video_capture():

    # opens video feed and then can be used to take an image for testing using SPACE
    # might want to change the gray varaible when we get aroudn to actually processing more complex data

    cap = cv.VideoCapture(0)
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
            cv.imwrite(f"test_images\{img_name}", gray)
            print(f"{img_name} written!")
            cv.imshow(f"{img_name}", gray)
            
            
    cap.release()
    cv.destroyAllWindows()
    




def grid_setup():

    img = np.array(cv.imread("test_images\\test_image.png"))
    fan_angle = np.pi()/4
    shape = img.shape
    if shape[0] <= shape[1]:
        ring_rad = (shape[1]*np.tan((np.pi)/2-fan_angle))/2 + shape[0]/2
    elif shape[0] > shape[1]:
        ring_rad = (shape[0]*np.tan((np.pi)/2-fan_angle))/2 + shape[1]/2
    