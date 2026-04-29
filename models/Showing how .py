import numpy as np
import cv2 as cv
import os

from fixed_model import fan_setup, ring_thing
from Noiseless import get_edge_sharpness, ssim, ART_solver


N = 64

# Run forward projection once at default parameters to get true_img
fan_list = fan_setup(np.pi/4, no_beams=96)
A, b, img = ring_thing(fan_list,
                       ring_subdivisions=180,
                       beam_subdivisions=100,
                       aperture=1,
                       image_string="phantom.png",
                       resize=N)

# Ground truth in log-space
x_true   = np.log(np.clip(
    cv.cvtColor(cv.resize(cv.imread("test_images/phantom.png"), (N, N)),
                cv.COLOR_BGR2GRAY).astype(float) / 255, 1e-6, None)).flatten()
true_img = x_true.reshape(N, N)

# Global scale
g_min      = np.min(true_img)
g_max      = np.max(true_img)
data_range = g_max - g_min
s_true, _  = get_edge_sharpness(true_img, g_min, g_max)