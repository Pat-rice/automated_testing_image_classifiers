__author__ = 'patrice'
import mongo_driver
from PIL import Image
import image_processing
import numpy as np
import matplotlib.pyplot as plt

def test_edges():
    """
    Function to experiment with the edge detection
    Load one image, detect edges and display it alongside the original image
    This function is not part of the main application
    :return:
    """
    raw_img = mongo_driver.load_one_and_show(8)
    try:
        image_read = Image.open(raw_img)
        #Detect edges
        img_edge = image_processing.detect_edges(np.array(image_read), True)

        # display results
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
        ax1.set_axis_off()
        ax1.imshow(img_edge, cmap=plt.cm.gray, interpolation='nearest')
        ax2.imshow(image_read, interpolation='nearest')
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(e)

test_edges()