from pymongo import Connection
import gridfs

# setup mongo
MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
DB_NAME = 'lsic' #TODO rename
COLLECTION_NAME_RAW_IMAGE = 'image'
COLLECTION_NAME_EDGES = 'edges'
COLLECTION_NAME_NORMALIZED_DATA = 'normalized_data'

# connect to the database & get a gridfs handle
mongo_con = Connection(MONGODB_HOST, MONGODB_PORT)

image_collection = gridfs.GridFS(mongo_con[DB_NAME], COLLECTION_NAME_RAW_IMAGE)
edges_collection = mongo_con[DB_NAME][COLLECTION_NAME_EDGES]
normalized_data = mongo_con[DB_NAME][COLLECTION_NAME_NORMALIZED_DATA]


def save_normalized_data(vector, target):
    d = {
        'vector': vector,
        'target': target
    }
    normalized_data.insert(d)


def get_normalized_data():
    return normalized_data.find().limit(9000)


def save_raw_image(raw_data, gridfs_filename, mime_type, category_name):
    # insert the resource into gridfs using the raw stream
    _id = image_collection.put(raw_data, contentType=mime_type, filename=gridfs_filename, category=category_name)
    return _id


def save_edges(edges_data, filename, category_name, raw_image_id):
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
    :return:
    """
    return image_collection.find()


def get_edges_from_category(category, limit):
    return edges_collection.find({'category': category}, {'edges_data': 1}).limit(limit)


def get_edges_exclude_category(category):
    return edges_collection.find({'category': {'$ne': category}}, {'edges_data': 1})


def get_categories_values():
    return edges_collection.distinct('category')


#TODO remove
def load_one_and_show(nb):
    list = image_collection.find()
    file_gridout = image_collection.get(list[nb]._id)
    return file_gridout



# def loadOneEdgeData():
#     doc = edges_collection.find_one()
#     print(type(doc))
#     return numpy.array(doc['edges_data'])