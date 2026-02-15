import numpy as np
import cv2 as cv

img = np.array(cv.imread("test_images\\test_image.png"))
cock = (img.shape)[:2]
print(cock)
