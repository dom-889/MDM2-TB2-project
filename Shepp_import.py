from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt

phantom = shepp_logan_phantom()

plt.imshow(phantom, cmap='gray')