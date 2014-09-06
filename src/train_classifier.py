import mongo_driver
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation, preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from skimage.transform import rescale

#conf
nbCategories = 6
limit_images_per_categories = 250

dataset_train = np.array([])
dataset_test = np.array([])
targets_train = np.array([])
targets_test = np.array([])
max_pixels = 40000
# max_pixels = 10000

categories_values = mongo_driver.get_categories_values()
i = 0
for ctg in categories_values:
    print("adding category : {}".format(ctg))
    all_edges_cursor = mongo_driver.get_edges_from_category(ctg, limit_images_per_categories)
    # all_edges_cursor = mongo_driver.get_full_edges_from_category(ctg, limit_images_per_categories)

    temp_dataset = np.array([])
    temp_targets = np.array([])

    for row in all_edges_cursor:
        #Scale the image edges to normalize it
        edges = np.array(row['edges_data'])
        edges = np.asfarray(edges)
        maxDim = np.max(edges.shape)
        scale = 200 / maxDim
        edges_scaled = rescale(edges, scale)
        # Flatten 2D edges to 1D vector
        pixels_vector = np.array(edges_scaled).flatten()
        # pixels_vector = np.array(edges).flatten()
        if pixels_vector.size < max_pixels:
            diff = max_pixels - pixels_vector.size
            #Fill up vector with false values to normalise images
            pixels_vector = np.concatenate([pixels_vector, [False] * diff])

        if len(temp_dataset) == 0:
            temp_dataset = pixels_vector
        else:
            temp_dataset = np.vstack((temp_dataset, pixels_vector))
        temp_targets = np.append(temp_targets, i)
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


resFile = open("../results/results.txt", "a")
start = time.time()

##################MODELS##########################

# Create and fit a nearest-neighbor classifier
# fast to train, slow to test, need good proportion of chosen cat vs others
# model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', n_neighbors=5, p=1, weights='uniform')

# from sklearn.naive_bayes import BernoulliNB
# model = BernoulliNB()

# from sklearn.svm import NuSVC
# model = NuSVC()


# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier(random_state=0)

# from sklearn.neural_network import BernoulliRBM
# from sklearn.pipeline import Pipeline
# from sklearn import linear_model
# logistic = linear_model.LogisticRegression(C=5000.0)
# logistic = linear_model.RidgeClassifierCV()
# rbm = BernoulliRBM(n_components=250, learning_rate=0.07, n_iter=15, verbose=1, random_state=None)
# rbm = BernoulliRBM()
# model = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

# from sklearn.linear_model import SGDClassifier
# model = SGDClassifier(loss="hinge", alpha=0.01, n_iter=200, fit_intercept=True)


###ENSEMBLE TESTS
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier

# model = RandomForestClassifier(n_estimators=50)
# model = GradientBoostingClassifier(n_estimators=30)
model = ExtraTreesClassifier(n_estimators=100)
# model = AdaBoostClassifier(n_estimators=30)

print('training...')
ovaM = OneVsRestClassifier(model, n_jobs=-1).fit(dataset_train, targets_train)
# ovaM = OneVsOneClassifier(model, n_jobs=-1).fit(dataset_train, targets_train)

training_end = time.time()
print('training ended in : {}'.format(training_end - start))

print('testing...')
#predict the dataset test
predictions = ovaM.predict(dataset_test)


totalTime = time.time() - start

# Compute confusion matrix
# cm = confusion_matrix(targets_test, predictions)

# print(cm)

# Show confusion matrix in a separate window
# plt.matshow(cm)
# plt.title('Confusion matrix')
# plt.colorbar()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()

from sklearn import metrics
print(metrics.classification_report(targets_test, predictions))

#Write results to file
resFile.write('#categories:{};'.format(nbCategories))
resFile.write('#imagesPerCat:{};'.format(limit_images_per_categories))
resFile.write('res:{}x{};\n'.format(np.sqrt(max_pixels), np.sqrt(max_pixels)))
resFile.write('strategy:{};\n'.format(ovaM))
resFile.write('model:{};\n'.format(model))
resFile.write('time:{};\n'.format(totalTime))
resFile.write('{}'.format(metrics.classification_report(targets_test, predictions)))
resFile.write('\n\n\n')
resFile.close()