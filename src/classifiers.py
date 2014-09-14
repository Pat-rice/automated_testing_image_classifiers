import mongo_driver
from sklearn import cross_validation
import numpy as np
import time
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import NuSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model


def run_models_comparison():
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
            5 : RidgeClassifierCV + BernoulliRBM
        max_pixels : maximum number of pixels for normalization

    """
    print('loading data in memory')
    max_number_image_per_category = 900
    in_memory_dataset = np.zeros(shape=(9000, 40000))
    in_memory_targets = np.zeros(9000)
    normalized_data = mongo_driver.get_normalized_data()
    idx = 0
    for nd in normalized_data:
        in_memory_dataset[idx] = np.array(nd['vector'])
        in_memory_targets[idx] = nd['targets']
        idx += 1
    print('finished loading')

    #for number of categories to classify
    for ix in range(1, 10):
        #for each model
        for jy in range(1, 6):
            #for each nb image
            for kz in range(1, 6):

                print('training nb category {}, model {}, nb image {}'.format(ix, jy, kz * 100))
                nbCategories = ix
                limit_images_per_categories = 100 * kz
                model_id = jy
                max_pixels = 40000

                #Get a subset of data from the dataset in memory
                index1 = int(nbCategories * max_number_image_per_category)
                index2 = int(max_number_image_per_category / limit_images_per_categories)
                temp_dataset = in_memory_dataset[:index1:index2]
                temp_targets = in_memory_targets[:index1:index2]

                #Split the subset into training and testing data
                dataset_train, dataset_test, targets_train, targets_test = cross_validation.train_test_split(temp_dataset, temp_targets, random_state=0)

                start = time.time()

                ##################MODELS##########################
                if model_id == 1:
                    model = RandomForestClassifier(n_estimators=50)
                elif model_id == 2:
                    model = AdaBoostClassifier(n_estimators=50)
                elif model_id == 3:
                    model = NuSVC()
                elif model_id == 4:
                    model = BernoulliNB()
                elif model_id == 5:
                    logistic = linear_model.RidgeClassifierCV()
                    rbm = BernoulliRBM()
                    model = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

                print('training...')
                try:
                    #Training using One vs all strategy, on all processor cores
                    ovaM = OneVsRestClassifier(model, n_jobs=-1).fit(dataset_train, targets_train)

                    print('testing...')
                    #Predict the dataset test
                    predictions = ovaM.predict(dataset_test)

                    totalTime = time.time() - start
                    print('overall execution time : {}'.format(totalTime))

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
                except Exception as e:
                    print(e)