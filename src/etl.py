__author__ = 'patrice'
import mongo_driver
import requests
import mimetypes
import os
import io
import numpy as np
import image_processing
from PIL import Image
from skimage.transform import rescale
import configparser

#Load config file
config = configparser.ConfigParser()
config.read('../project.cfg')


def load_images_from_urls():
    """
    Read list of urls from resources folder, try to download every image and save it to the database
    :return:
    """
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
                    print('not an image : {}'.format(url))
                    print(e)
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
            img_edge = image_processing.detect_edges(np.array(image_read))
            #Save it to DB
            mongo_driver.save_edges(np.array(img_edge).tolist(), raw_img.filename, raw_img.category, raw_img._id)
            total_ok += 1
        except Exception as e:
            print(e)
            total_notok += 1
            continue
    print('Total edges saved : {}'.format(total_ok))
    print('Total errors : {}'.format(total_notok))


def normalize_dataset():
    max_pixels = config.getint('image_processing', 'max_number_normalised_pixels')
    images_per_category = config.getint('classifiers', 'max_number_images_per_category')
    max_side_size = config.getint('image_processing', 'max_side_size')
    all_categories = mongo_driver.get_categories_values()
    i = 0
    for category in all_categories:
        print("adding category : {}".format(category))
        all_edges_cursor = mongo_driver.get_edges_from_category(category, images_per_category)

        for row in all_edges_cursor:

            #Scale the image edges to normalize it
            edges = np.array(row['edges_data'])
            edges = np.asfarray(edges)
            max_dim = np.max(edges.shape)
            scale = max_side_size / max_dim
            edges_scaled = rescale(edges, scale)

            # Flatten 2D edges to 1D vector
            pixels_vector = np.array(edges_scaled).flatten()

            if pixels_vector.size < max_pixels:
                diff = max_pixels - pixels_vector.size
                #Fill up vector with false values to normalise images
                pixels_vector = np.concatenate([pixels_vector, [False] * diff])

            #Save it to DB
            mongo_driver.save_normalized_data(np.array(pixels_vector).tolist(), i)
        i += 1

