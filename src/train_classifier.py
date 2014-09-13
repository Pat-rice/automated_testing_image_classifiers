import mongo_driver
from sklearn import cross_validation, preprocessing
import numpy as np
import time
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from skimage.transform import rescale
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import NuSVC, SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model


def load_edges_data_in_memory():
    dataset = np.array([])
    targets = np.array([])
    all_categories = mongo_driver.get_categories_values()
    i =0
    for category in all_categories:
        print("adding category : {}".format(category))
        all_edges_cursor = mongo_driver.get_edges_from_category(category, 10)

        for row in all_edges_cursor:
            #Scale the image edges to normalize it
            edges = np.array(row['edges_data'])
            edges = np.asfarray(edges)
            max_dim = np.max(edges.shape)
            scale = 200 / max_dim
            edges_scaled = rescale(edges, scale)
            # Flatten 2D edges to 1D vector
            pixels_vector = np.array(edges_scaled).flatten()
            # pixels_vector = np.array(edges).flatten()
            if pixels_vector.size < max_pixels:
                diff = max_pixels - pixels_vector.size
                #Fill up vector with false values to normalise images
                pixels_vector = np.concatenate([pixels_vector, [False] * diff])

            if len(dataset) == 0:
                dataset = pixels_vector
            else:
                dataset = np.vstack((dataset, pixels_vector))
            targets = np.append(targets, i)
    return dataset, targets


# def loadEdgesFromMemory(category_name, nb_images):




""" Configuration of classifier builder
    Parameters
    ----------
    nbCategories : integer
        the number of categories that will be used
    limit_images_per_categories : integer
        the number of images per category that will be used
    algorithm_id : integer
        if of the algorithm used
        1 : RandomForestClassifier
        2 : AdaBoostClassifier
        3 : NuSVC4
        4 : BernoulliNB
    max_pixels : maximum number of pixels for normalization

"""

in_memory_dataset, in_memory_targets = load_edges_data_in_memory()
#for nb category
for ix in range(2, 10):
    #for each model
    for jy in range(1, 5):
        #for each nb image
        for kz in range(1, 11):

            print('training nb category {}, model {}, nb image {}'.format(ix, jy, kz))
            nbCategories = ix
            limit_images_per_categories = 100 * kz
            model_id = jy
            max_pixels = 40000

            dataset_train = np.array([])
            dataset_test = np.array([])
            targets_train = np.array([])
            targets_test = np.array([])

            categories_values = mongo_driver.get_categories_values()
            i = 0
            for ctg in categories_values:


                # print("adding category : {}".format(ctg))
                # all_edges_cursor = mongo_driver.get_edges_from_category(ctg, limit_images_per_categories)
                #
                # temp_dataset = np.array([])
                # temp_targets = np.array([])
                #
                # for row in all_edges_cursor:
                #     #Scale the image edges to normalize it
                #     edges = np.array(row['edges_data'])
                #     edges = np.asfarray(edges)
                #     maxDim = np.max(edges.shape)
                #     scale = 200 / maxDim
                #     edges_scaled = rescale(edges, scale)
                #     # Flatten 2D edges to 1D vector
                #     pixels_vector = np.array(edges_scaled).flatten()
                #     # pixels_vector = np.array(edges).flatten()
                #     if pixels_vector.size < max_pixels:
                #         diff = max_pixels - pixels_vector.size
                #         #Fill up vector with false values to normalise images
                #         pixels_vector = np.concatenate([pixels_vector, [False] * diff])
                #
                #     if len(temp_dataset) == 0:
                #         temp_dataset = pixels_vector
                #     else:
                #         temp_dataset = np.vstack((temp_dataset, pixels_vector))
                #     temp_targets = np.append(temp_targets, i)

                x = int(nbCategories * 1000)
                y = int(1000/limit_images_per_categories)
                temp_dataset = in_memory_dataset[:x:y]

                i += 1
                # Split the data into a training set and a test set
                d_train, d_test, t_train, t_test = cross_validation.train_test_split(temp_dataset, temp_targets, random_state=0)
                if len(dataset_train) == 0:
                    dataset_train = d_train
                    dataset_test = d_test
                    targets_train = t_train
                    targets_test = t_test
                else:
                    dataset_train = np.vstack((dataset_train, d_train))
                    dataset_test = np.vstack((dataset_test, d_test))
                    targets_train = np.append(targets_train, t_train)
                    targets_test = np.append(targets_test, t_test)

                if i == nbCategories:
                    break

            start = time.time()

            ##################MODELS##########################
            if model_id == 1:
                model = RandomForestClassifier(n_estimators=50)
            elif model_id == 2:
                model = AdaBoostClassifier(n_estimators=30)
            elif model_id == 3:
                model = NuSVC()
            elif model_id == 4:
                model = BernoulliNB()
            # elif model_id == 5:
            #     logistic = linear_model.RidgeClassifierCV()
            #     rbm = BernoulliRBM()
            #     model = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

            print('training...')
            #Training using One vs all strategy
            ovaM = OneVsRestClassifier(model, n_jobs=-1).fit(dataset_train, targets_train)
            # ovaM = OneVsOneClassifier(model, n_jobs=-1).fit(dataset_train, targets_train)

            training_end = time.time()
            print('training ended in : {}'.format(training_end - start))

            print('testing...')
            #Predict the dataset test
            predictions = ovaM.predict(dataset_test)

            totalTime = time.time() - start

            from sklearn import metrics
            print(metrics.classification_report(targets_test, predictions))



            #Write results to file
            resFile = open("../results/results.txt", "a")
            resFile.write('#categories:{};'.format(nbCategories))
            resFile.write('#imagesPerCat:{};'.format(limit_images_per_categories))
            resFile.write('res:{}x{};\n'.format(np.sqrt(max_pixels), np.sqrt(max_pixels)))
            resFile.write('strategy:{};\n'.format(ovaM))
            resFile.write('model:{};\n'.format(model))
            resFile.write('time:{};\n'.format(totalTime))
            resFile.write('{}'.format(metrics.classification_report(targets_test, predictions)))
            resFile.write('\n\n\n')
            resFile.close()

            shortResFile = open("../results/average_results.csv", "a")
            shortResFile.write('{};'.format(nbCategories))
            shortResFile.write('{};'.format(limit_images_per_categories))
            shortResFile.write('{}x{};'.format(np.sqrt(max_pixels), np.sqrt(max_pixels)))
            shortResFile.write('{};'.format(model_id))
            shortResFile.write('{};'.format(metrics.precision_score(targets_test, predictions)))
            shortResFile.write('{};'.format(metrics.recall_score(targets_test, predictions)))
            shortResFile.write('{};'.format(metrics.f1_score(targets_test, predictions)))
            shortResFile.write('{};\n'.format(totalTime))
            shortResFile.close()