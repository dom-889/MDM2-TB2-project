from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt

phantom = shepp_logan_phantom()

plt.figure()
plt.imshow(phantom, cmap='gray')
plt.axis('off') 
plt.savefig("test_images/shepp_logan_phantom.png", bbox_inches='tight', pad_inches=0)
plt.close()  # Close figure to avoid overlap

plt.figure(figsize=(2, 2), dpi=32)  # 2*32 = 64 pixels
plt.imshow(phantom, cmap='gray')
plt.axis('off') 
plt.savefig("test_images/shepp_logan_phantom64x64.png", bbox_inches='tight', pad_inches=0)
plt.close()


'''
plt.imshow(phantom, cmap='gray')
plt.axis('off') 
plt.savefig("test_images/shepp_logan_phantom.png", bbox_inches='tight', pad_inches=0)

plt.figure(figsize=(2, 2), dpi=32)
plt.imshow(phantom, cmap='gray')
plt.axis('off') 
plt.savefig("test_images/shepp_logan_phantom64x64.png", bbox_inches='tight', pad_inches=0)

plt.show()
'''