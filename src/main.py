__author__ = 'patrice'

import etl
import classifiers


def run_flow(flow_id):
    """ Start an application flow depending on the flow_id
        Parameters
        ----------
        flow_id : integer
            id of the flow to run
            1 : download dataset from list of urls and save it to database
            2 : load original images from database, find edges and save it to database
            3 : load edges from database, normalize it and save it to database
            4 : load normalize data, build models with different configuration and increase the load, save the results to an external file.
    """
    if flow_id == 1:
        etl.load_images_from_urls()
    elif flow_id == 2:
        etl.find_edges_and_save()
    elif flow_id == 3:
        etl.normalize_dataset()
    elif flow_id == 4:
        classifiers.run_models_comparison()

#Start flow
run_flow(1)
