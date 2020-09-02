# import random
import time

import numpy as np
import scipy.io as sio
from scipy import stats
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import svm

from iForest import iForest
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn import cluster
# from sklearn import preprocessing
# from sklearn.metrics import roc_auc_score


def cross_validation(X, y, ratio=.7):
    # cross validation
    X_train, X_test, y_train, y_test = \
        model_selection.train_test_split(
            X, y, test_size=(1-ratio), shuffle=True)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
    }


def load_from_mat(mat_fname):
    mat_contents = sio.loadmat(mat_fname)
    X = mat_contents['X']
    if(mat_fname == 'mnist.mat'):
        # has many attribute with no information
        # remove these attributes here
        # has 76/100 attributes remaining
        X = X[:, [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27,
                  29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51,
                  54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 71, 73, 74, 75, 76, 77,
                  79, 80, 81, 82, 83, 84, 85, 90, 92, 93, 94, 95, 96, 97, 98]]
    y = mat_contents['y'].astype(int).ravel()
    y = 1 - y
    return X, y


def main():
    # datasets = {'cardio.mat','shuttle.mat'}
    # datasets = {'thyroid.mat'}
    datasets = {'mnist.mat'}
    print('================================================================')
    print('> Loading Data ...')
    for mat_fname in datasets:
        X, y = load_from_mat(mat_fname)
        X = stats.zscore(X)
        print('> dataset: ', mat_fname)
        print('================================================================')
        algorithms = {
            # 'KNN': neighbors.KNeighborsClassifier(),
            'SVM': svm.SVC(gamma='auto'),
            # 'SVM_linear': svm.OneClassSVM(kernel='linear'),
            # 'RandomForest': ensemble.RandomForestClassifier(),
            # 'ExtraTrees': ensemble.ExtraTreesClassifier(n_estimators=100),
            # 'DBSCAN': cluster.DBSCAN(eps=1, min_samples=5),
            # 'iForest_create': iForest(with_replacement=True),
            'OneClassSVM': svm.OneClassSVM(gamma='auto'),
            'iForest_n': iForest(with_replacement=True, project_flag=True),
            'iForest_create': iForest(with_replacement=True, project_flag=False),
            # 'iForest_paper': iForest(with_replacement=False),
            'IsolationForest': ensemble.IsolationForest(contamination=0.2, behaviour='new'),
            'LOF': neighbors.LocalOutlierFactor(contamination=0.2, novelty=True),
        }
        print('> Algorithms: {}'.format([i for i, j in algorithms.items()]))
        print('> Accuracy for iForest is not correct')
        for name, clf in algorithms.items():
            print('---------------------------------------------')
            print('> {}'.format(name))
            sum_accuracy = 0
            auc_value = []
            total_time = 0
            ratio = 0.7
            iteration = 5

            for i in range(iteration):
                data = cross_validation(X, y, ratio)
                t0 = time.process_time()
                if(  # unsupervised method
                    name == 'IsolationForest' or
                    name == 'OneClassSVM' or
                    name == 'LOF' or name == 'iForest_n' or
                    name == 'iForest_create' or name == 'iForest_paper'
                ):
                    X_train = np.array(data['X_train'])
                    y_train = np.array(data['y_train'])
                    X_train = X_train[np.where(y_train == 1)]
                    clf.fit(X_train)
                else:
                    # supervised method.
                    clf.fit(data['X_train'], data['y_train'])

                t1 = time.process_time()
                y_pred = clf.predict(data['X_test'])
                # if(name != 'KNN' and name != 'RandomForest'):
                # y_score = clf.decision_function(data['X_test'])

                y_real = data['y_test']
                y_real = np.array(y_real)
                y_pred[np.where(y_pred == -1)] = 0
                if(  # methods with decision functions and have ROC curves
                    name == 'SVM' or name == 'SVM_linear' or name == 'LOF'
                    or name == 'OneClassSVM' or name == 'IsolationForest'
                    or name == 'iForest_n' or
                    name == 'iForest_create' or name == 'iForest_paper'
                ):
                    y_score = np.asarray(clf.decision_function(data['X_test']))
                    if(name == 'iForest_n' or name == 'iForest_create'):
                        y_score = -1 * y_score
                    fpr, tpr, thresholds = metrics.roc_curve(data['y_test'],
                                                             y_score,
                                                             pos_label=1)
                    roc_auc = metrics.auc(fpr, tpr)
                    auc_value.append(roc_auc)
                total_time += (t1-t0)
            print('\nAUC value is: ', np.mean(np.asarray(auc_value)))
            print('Accuracy: {}'.format(float(sum_accuracy)/iteration))
            print('Training Time: {}'.format(float(total_time)/iteration))
        print('---------------------------------------------')
        print('==============================================================')


if __name__ == '__main__':
    main()
