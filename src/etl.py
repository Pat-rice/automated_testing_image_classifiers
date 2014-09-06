__author__ = 'patrice'
import mongo_driver
import requests
import mimetypes
import os
import io
import numpy as np
import edge_detection
from PIL import Image


def load_images_from_urls():
    total_added = 0
    #List of the urls files
    list_files = os.listdir('../resources')
    for file in list_files:
        if 'urls' in file:
        # if 'samples' in file:
            print('reading {} ...'.format(file))
            file_stream = open('../resources/{}'.format(file), 'r')
            for url in file_stream:
                url = url.rstrip()
                try:
                    r = requests.get(url, stream=True, timeout=10)
                    #Get the response stream
                    stream = io.BytesIO(r.raw.data)
                    #Try to open as Image to check the validity of the url (could be a document not found)
                    Image.open(stream)
                    # guess the mimetype and request the image resource
                    mime_type = mimetypes.guess_type(url)[0]
                    # save raw data to DB
                    mongo_driver.save_raw_image(r.raw.data, url, mime_type, file)
                    total_added += 1
                except Exception as e:
                    # print('not an image : {}'.format(url))
                    # print(e)
                    continue

            print('end {}'.format(file))
    print('{} images have been added to the database !'.format(total_added))


def find_edges_and_save():
    total_ok = 0
    total_notok = 0
    all_images = mongo_driver.get_all_raw_images()
    for raw_img in all_images:
        try:
            image_read = Image.open(raw_img)
            #Detect edges
            img_edge = edge_detection.detect_edges(np.array(image_read))
            ##TODO add matrix compression
            #Save it to DB
            # mongo_driver.save_edges(np.array(img_edge).tolist(), raw_img.filename, raw_img.category, raw_img._id)
            mongo_driver.save_full_edges(np.array(img_edge).tolist(), raw_img.filename, raw_img.category, raw_img._id)
            total_ok += 1
        except Exception as e:
            print(e)
            total_notok += 1
            continue
    print(total_ok)
    print(total_notok)

find_edges_and_save()




import matplotlib.pyplot as plt
import io
def test_edges():
    raw_img = mongo_driver.load_one_and_show(6000)
    try:
        image_read = Image.open(raw_img)
        #Detect edges
        img_edge = edge_detection.detect_edges(np.array(image_read))

        # display results
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
        ax1.imshow(img_edge, cmap=plt.cm.gray)
        ax2.imshow(image_read)
        plt.show()

    except Exception as e:
        print(e)

# test_edges()

