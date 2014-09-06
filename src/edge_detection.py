__author__ = 'patrice'
from skimage import filter
from skimage.color import rgb2gray
from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import binary_dilation, diamond
from skimage.segmentation import clear_border


def detect_edges(image_array):
    """ Detect edges in a given image
    Takes a numpy.array representing an image, apply edges detection and return a numpy.array
    :param image_array:
    :return:
    """
    #Transform image into grayscale
    img = rgb2gray(image_array)
    #Remove some noise from the image
    img = denoise_tv_chambolle(img, weight=0.55)
    #Apply canny
    edges = filter.canny(img, sigma=3.2)
    #Clear the borders
    clear_border(edges, 15)
    #Dilate edges to make them more visible and connected
    edges = binary_dilation(edges, selem=diamond(3))
    return edges


