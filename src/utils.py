__author__ = 'patrice'
import mongo_driver
from PIL import Image
import image_processing
import numpy as np
import matplotlib.pyplot as plt
import io
def test_edges():
    raw_img = mongo_driver.load_one_and_show(6000)
    try:
        image_read = Image.open(raw_img)
        #Detect edges
        img_edge = image_processing.detect_edges(np.array(image_read))

        # display results
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
        ax1.imshow(img_edge, cmap=plt.cm.gray)
        ax2.imshow(image_read)
        plt.show()

    except Exception as e:
        print(e)

test_edges()