from pymongo import Connection
import gridfs
import configparser

#Load config file
config = configparser.ConfigParser()
config.read('../project.cfg')

# setup mongo
MONGODB_HOST = config.get('database', 'host')
MONGODB_PORT = config.getint('database', 'port')
DB_NAME = config.get('database', 'name')
COLLECTION_NAME_RAW_IMAGE = config.get('database', 'collection_name_original_image')
COLLECTION_NAME_EDGES = config.get('database', 'collection_name_edges')
COLLECTION_NAME_NORMALIZED_DATA = config.get('database', 'collection_name_normalized_data')

# connect to the database & get a gridfs handle
mongo_con = Connection(MONGODB_HOST, MONGODB_PORT)

image_collection = gridfs.GridFS(mongo_con[DB_NAME], COLLECTION_NAME_RAW_IMAGE)
edges_collection = mongo_con[DB_NAME][COLLECTION_NAME_EDGES]
normalized_data = mongo_con[DB_NAME][COLLECTION_NAME_NORMALIZED_DATA]


def save_normalized_data(vector, target):
    """
    Save normalized data to the database
    :param vector: numpy array
        represents a flatten normalized version of the edge matrix
    :param target: integer
        category id associated the vector
    """
    d = {
        'vector': vector,
        'target': target
    }
    normalized_data.insert(d)


def get_normalized_data():
    """
    Returns a dataset of normalized data
    :return: mongo cursor
    """
    total_images = config.getint('classifiers', 'total_images')
    return normalized_data.find().limit(total_images)


def save_raw_image(raw_data, gridfs_filename, mime_type, category_name):
    """
    Save original image into a gridfs collection
    :param raw_data:
        original image data as stream
    :param gridfs_filename:
        name of the image
    :param mime_type:
        type of the image
    :param category_name:
        name of the image category
    :return:
    """
    _id = image_collection.put(raw_data, contentType=mime_type, filename=gridfs_filename, category=category_name)
    return _id


def save_edges(edges_data, filename, category_name, raw_image_id):
    """
    Save edges to the database
    :param edges_data:
        boolean matrix representing edges of an image
    :param filename:
        image name
    :param category_name:
        image category name
    :param raw_image_id:
        mongo object id of the original image
    :return:
    """
    tmp = {
        'edges_data': edges_data,
        'filename': filename,
        'category': category_name,
        'rawImageId': raw_image_id
    }
    _edges_id = edges_collection.insert(tmp)
    return _edges_id


def get_all_raw_images():
    """ Get all raw images
    Return a list of gridfs
    :return mongo cursor
    """
    return image_collection.find()


def get_edges_from_category(category, limit):
    """
    Get edges data from a given category
    :param category:
        category name
    :param limit:
        maximum number of edges to return
    :return: mongo cursor
    """
    return edges_collection.find({'category': category}, {'edges_data': 1}).limit(limit)


def get_categories_values():
    """
    Get all categories name available in the database
    :return: mongo cursor
    """
    return edges_collection.distinct('category')


def load_one_and_show(nb):
    """
    Load one image and returns it,
    this function is used for experiment only and is not part of the main application
    :param nb:
    :return:
    """
    list = image_collection.find()
    file_gridout = image_collection.get(list[nb]._id)
    return file_gridout
