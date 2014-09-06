import mongo_driver
from sklearn import cross_validation
import numpy as np

limit_categories = 200
dataset = np.array([])
targets = np.array([])
max_pixels = 10000

categories_values = mongo_driver.get_categories_values()
i = 0
for cat in categories_values:
    print("training : {}".format(cat))
    # if chosen_category not in cat:
    all_edges_cursor = mongo_driver.get_edges_from_category(cat, limit_categories)

    for image in all_edges_cursor:
        pixels_vector = np.array(image['edges_data']).flatten()
        if pixels_vector.size < max_pixels:
            diff = max_pixels - pixels_vector.size
            #Fill up vector with false values to normalise images
            pixels_vector = np.concatenate([pixels_vector, [False] * diff])

        if len(dataset) == 0:
            dataset = pixels_vector
        else:
            dataset = np.vstack((dataset, pixels_vector))
        targets = np.append(targets, i)
    i += 1

dataset_train, dataset_test, targets_train, targets_test = cross_validation.train_test_split(dataset, targets, random_state=0)

from sklearn import grid_search
from sklearn.svm import NuSVC
from sklearn.ensemble import BaggingClassifier
# model = NuSVC(nu=0.7, kernel='sigmoid', degree=3, gamma=0.2, coef0=0.1, shrinking=True, probability=True, tol=0.001, cache_size=200, max_iter=-1, random_state=None)
# model = SVC()
svc = NuSVC()
model = BaggingClassifier()
parameters = {
    'kernel':('linear', 'rbf', 'sigmoid'),
    'gamma':[0.1, 0.2, 0.3, 0.4],
    'coef0':[0.05, 0.1, 0.15],
    'nu':[0.4, 0.5, 0.6, 0.7, 0.8],
    # 'degree' : [0,1,2,3,4,5,6],
    # 'shrinking' : [True, False],
    'probability' : [True, False],
    # 'tol' : [0.001, 0.0001, 0.01]
}
parameters = {
    'n_estimators' : [10,20,30,40,50,60]
}

clf = grid_search.GridSearchCV(model, parameters, n_jobs=-1)
clf.fit(dataset_train, targets_train)

# print(clf.grid_scores_)
# print(clf.best_estimator_)
print(clf.best_params_)
print(clf.best_score_)