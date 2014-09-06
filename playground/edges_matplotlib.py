import numpy as np
import matplotlib.pyplot as plt
import mongo_driver
from skimage import data
from PIL import Image
from skimage.color import rgb2gray
from skimage import morphology
from skimage.filter import sobel, threshold_otsu
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman
from skimage.transform import resize, rescale

#img 1900 good example before/after denoise

raw_img = mongo_driver.load_one_and_show(6000)
image = rgb2gray(np.array(Image.open(raw_img)))
image = denoise_tv_chambolle(image, weight=0.55)

from skimage.filter import canny
edges = canny(image, sigma=3.2)

#TODO Actually remove the border, less useless pixels to keep in vector
from skimage.segmentation import clear_border
clear_border(edges, 15)

from skimage.morphology import binary_dilation, diamond
image_cleaned = binary_dilation(edges, selem=diamond(3))

maxDim = np.max(image_cleaned.shape)
scale =  200 / maxDim
image_cleaned = rescale(image_cleaned, scale)


# coins = data.coins()
#
#
# hist = np.histogram(image, bins=np.arange(0, 20))
# elevation_map = sobel(image)
# markers = np.zeros_like(image)
# markers[image < 30] = 1
# markers[image > 150] = 2
# segmentation = morphology.watershed(elevation_map, markers)

# fig, img1  = plt.subplots(1, 1, figsize=(20, 20))
# img1.plot(hist[1][:-1], hist[0], lw=2)
# plt.show()

fig, (img1, img3, img4)  = plt.subplots(1, 3, figsize=(12, 5))
img1.imshow(edges, cmap=plt.cm.gray,  interpolation='nearest')
img3.imshow(image_cleaned, cmap=plt.cm.gray, interpolation='nearest')


# img1.imshow(elevation_map, cmap=plt.cm.jet, interpolation='nearest')
# img2.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
# img3.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
####original image
img4.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
img4.axis('off')
plt.show()