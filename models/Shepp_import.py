from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt

phantom = shepp_logan_phantom()

plt.figure()
plt.imshow(phantom, cmap='gray')
plt.axis('off') 
plt.savefig("test_images/shepp_logan_phantom.png", bbox_inches='tight', pad_inches=0)
plt.close() 

