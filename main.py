import warnings
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from scipy.io import loadmat
from imondrian.mondrianforest import MondrianForest
from imondrian.mondrianforest import process_command_line
from imondrian.mondrianforest_utils import load_data
from imondrian.mondrianforest_utils import precompute_minimal
from imondrian.mondrianforest_utils import reset_random_seed
from mondrianforest_demo import demo
warnings.filterwarnings('ignore')

from sklearn.datasets import make_moons, make_blobs, samples_generator
import os
import pickle
from sklearn.ensemble import IsolationForest  #--> https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest
from sklearn.model_selection import StratifiedShuffleSplit
import time
from sklearn.svm import OneClassSVM  #--> https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM
from sklearn.neighbors import LocalOutlierFactor as LOF  #--> https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope  #--> https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html#sklearn.covariance.EllipticEnvelope
import h5py
import pandas as pd
from sklearn.model_selection import KFold
from my_osPCA import My_osPCA
from my_online_LOF import My_online_LOF


def main():
    # settings:
    experiment = 5
    dataset = "smtp"  # --> two_moons, one_blob, two_blobs, two_different_blobs, Breast_cancer, Pima, Speech, Thyroid,
                            # Satellite, optdigits, letter, arrhythmia, ionosphere, http, shuttle, wine, annthyroid, smtp, musk, cardio, vowels, lympho
    method = "iMondrian_forest"  # --> iso_forest, one_class_SVM, LOF, covariance_estimator, iMondrian_forest, osPCA_powerMethod, osPCA_leastSquares, online_LOF
    generate_synthetic_datasets_again = False
    predict_anomaly_again = False
    predict_using_threshold = False  #--> True: by threshold, False: by K-means
    read_the_dataset = True
    split_in_cross_validation_again = False
    split_data_to_stages_again = False

    # functions:
    if read_the_dataset:
        X_train, Y_train, X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds = read_dataset(dataset_name=dataset, split_in_cross_validation_again=split_in_cross_validation_again)
        Y_train, y_train_in_folds, y_test_in_folds = convert_labels_to_1_and_minus_1(dataset=dataset, y=Y_train, y_train_in_folds=y_train_in_folds, y_test_in_folds=y_test_in_folds)
    if experiment == 1:
        experiment_original_code()
    elif experiment == 2:
        anomaly_visual_experiment_bach(dataset=dataset, method=method, generate_synthetic_datasets_again=generate_synthetic_datasets_again, predict_anomaly_again=predict_anomaly_again, predict_using_threshold=predict_using_threshold)
    elif experiment == 3:
        anomaly_visual_experiment_online(dataset=dataset, method=method, generate_synthetic_datasets_again=generate_synthetic_datasets_again, predict_anomaly_again=predict_anomaly_again, predict_using_threshold=predict_using_threshold)
    elif experiment == 4:
        anomaly_detection_AUC_experiment_batch(anomaly_method=method, dataset=dataset, X_train_in_folds=X_train_in_folds, X_test_in_folds=X_test_in_folds, y_train_in_folds=y_train_in_folds, y_test_in_folds=y_test_in_folds)
    elif experiment == 5:
        path_to_save = "./datasets/" + dataset + "/stages/"
        if split_data_to_stages_again:
            # split dataset for online stages:
            n_online_stages = 5
            X_stages, y_stages = stratified_split_of_data(X=X_train, y=Y_train, n_online_stages=n_online_stages)
            save_variable(X_stages, 'X_stages', path_to_save=path_to_save)
            save_variable(y_stages, 'y_stages', path_to_save=path_to_save)
        else:
            file = open(path_to_save + 'X_stages.pckl', 'rb')
            X_stages = pickle.load(file)
            file.close()
            file = open(path_to_save + 'y_stages.pckl', 'rb')
            y_stages = pickle.load(file)
            file.close()
        anomaly_detection_AUC_experiment_online(anomaly_method=method, dataset=dataset, X_stages=X_stages, y_stages=y_stages)


def read_dataset(dataset_name, split_in_cross_validation_again):
    if dataset_name == 'Breast_cancer':
        path_dataset = "./datasets/Breast_cancer/"
        data = pd.read_csv(path_dataset+"wdbc_data.txt", sep=",", header=None)  # read text file using pandas dataFrame: https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas
        labels_of_classes = ['M', 'B']
        X, y = read_BreastCancer_dataset(data=data, labels_of_classes=labels_of_classes)
        X = X.astype(np.float64)  # ---> otherwise MDS has error --> https://stackoverflow.com/questions/16990996/multidimensional-scaling-fitting-in-numpy-pandas-and-sklearn-valueerror
    elif dataset_name == "Pima":
        path_dataset = "./datasets/Pima/"
        data = loadmat(path_dataset + "pima.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "Speech":
        path_dataset = "./datasets/Speech/"
        data = loadmat(path_dataset + "speech.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "Thyroid":
        path_dataset = "./datasets/Thyroid/"
        data = loadmat(path_dataset + "thyroid.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "Satellite":
        path_dataset = "./datasets/Satellite/"
        data = loadmat(path_dataset + "satellite.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "optdigits":
        path_dataset = "./datasets/optdigits/"
        data = loadmat(path_dataset + "optdigits.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "letter":
        path_dataset = "./datasets/letter/"
        data = loadmat(path_dataset + "letter.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "arrhythmia":
        path_dataset = "./datasets/letter/"
        data = loadmat(path_dataset + "letter.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "ionosphere":
        path_dataset = "./datasets/ionosphere/"
        data = loadmat(path_dataset + "ionosphere.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "http":
        path_dataset = "./datasets/http/"
        with h5py.File(path_dataset+'http.mat', 'r') as f:
            a = list(f['X'])
            b = list(f['y'])
        dimension0 = a[0].reshape((-1, 1))
        dimension1 = a[1].reshape((-1, 1))
        dimension2 = a[2].reshape((-1, 1))
        X = np.column_stack((dimension0, dimension1))
        X = np.column_stack((X, dimension2))
        y = b[0]
        y = y.astype(int)
    elif dataset_name == "shuttle":
        path_dataset = "./datasets/ionosphere/"
        data = loadmat(path_dataset + "ionosphere.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "wine":
        path_dataset = "./datasets/wine/"
        data = loadmat(path_dataset + "wine.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
        X = X.astype(np.float64)
    elif dataset_name == "annthyroid":
        path_dataset = "./datasets/annthyroid/"
        data = loadmat(path_dataset + "annthyroid.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "smtp":
        path_dataset = "./datasets/smtp/"
        with h5py.File(path_dataset + 'smtp.mat', 'r') as f:
            a = list(f['X'])
            b = list(f['y'])
        dimension0 = a[0].reshape((-1, 1))
        dimension1 = a[1].reshape((-1, 1))
        dimension2 = a[2].reshape((-1, 1))
        X = np.column_stack((dimension0, dimension1))
        X = np.column_stack((X, dimension2))
        y = b[0]
        y = y.astype(int)
    elif dataset_name == "musk":
        path_dataset = "./datasets/musk/"
        data = loadmat(path_dataset + "musk.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "cardio":
        path_dataset = "./datasets/cardio/"
        data = loadmat(path_dataset + "cardio.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "vowels":
        path_dataset = "./datasets/vowels/"
        data = loadmat(path_dataset + "vowels.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "lympho":
        path_dataset = "./datasets/lympho/"
        data = loadmat(path_dataset + "lympho.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    # --- cross validation:
    path_to_save = path_dataset + "/CV/"
    number_of_folds = 10
    if split_in_cross_validation_again:
        train_indices_in_folds, test_indices_in_folds, X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds = cross_validation(X=X, y=y, n_splits=number_of_folds)
        save_variable(train_indices_in_folds, 'train_indices_in_folds', path_to_save=path_to_save)
        save_variable(test_indices_in_folds, 'test_indices_in_folds', path_to_save=path_to_save)
        save_variable(X_train_in_folds, 'X_train_in_folds', path_to_save=path_to_save)
        save_variable(X_test_in_folds, 'X_test_in_folds', path_to_save=path_to_save)
        save_variable(y_train_in_folds, 'y_train_in_folds', path_to_save=path_to_save)
        save_variable(y_test_in_folds, 'y_test_in_folds', path_to_save=path_to_save)
    else:
        file = open(path_to_save + 'train_indices_in_folds.pckl', 'rb')
        train_indices_in_folds = pickle.load(file)
        file.close()
        file = open(path_to_save + 'test_indices_in_folds.pckl', 'rb')
        test_indices_in_folds = pickle.load(file)
        file.close()
        file = open(path_to_save + 'X_train_in_folds.pckl', 'rb')
        X_train_in_folds = pickle.load(file)
        file.close()
        file = open(path_to_save + 'X_test_in_folds.pckl', 'rb')
        X_test_in_folds = pickle.load(file)
        file.close()
        file = open(path_to_save + 'y_train_in_folds.pckl', 'rb')
        y_train_in_folds = pickle.load(file)
        file.close()
        file = open(path_to_save + 'y_test_in_folds.pckl', 'rb')
        y_test_in_folds = pickle.load(file)
        file.close()
    return X, y, X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds

def convert_labels_to_1_and_minus_1(dataset, y, y_train_in_folds, y_test_in_folds):
    n_folds = len(y_train_in_folds)
    for fold_index in range(n_folds):
        y_train = y_train_in_folds[fold_index]
        y_test = y_test_in_folds[fold_index]
        if dataset == "Breast_cancer":
            y_train[y_train == 0] = -1
            y_test[y_test == 0] = -1
        if dataset == "Pima" or dataset == "Speech" or dataset == "Thyroid" or dataset == "Satellite" or dataset == "optdigits" or \
                dataset == "letter" or dataset == "arrhythmia" or dataset == "ionosphere" or dataset == "http" or dataset == "shuttle" or dataset == "wine" or \
                dataset == "annthyroid" or dataset == "smtp" or dataset == "musk" or dataset == "cardio" or dataset == "vowels" or dataset == "lympho":
            y_train[y_train == 1] = -1
            y_test[y_test == 1] = -1
            y_train[y_train == 0] = 1
            y_test[y_test == 0] = 1
        y_train_in_folds[fold_index] = y_train
        y_test_in_folds[fold_index] = y_test
    if dataset == "Breast_cancer":
        y[y == 0] = -1
        y[y == 0] = -1
    if dataset == "Pima" or dataset == "Speech" or dataset == "Thyroid" or dataset == "Satellite" or dataset == "optdigits" or \
            dataset == "letter" or dataset == "arrhythmia" or dataset == "ionosphere" or dataset == "http" or dataset == "shuttle" or dataset == "wine" or \
            dataset == "annthyroid" or dataset == "smtp" or dataset == "musk" or dataset == "cardio" or dataset == "vowels" or dataset == "lympho":
        y[y == 1] = -1
        y[y == 0] = 1
    return y, y_train_in_folds, y_test_in_folds

def anomaly_detection_AUC_experiment_batch(anomaly_method, dataset, X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds):
    rng = np.random.RandomState(42)
    n_folds = len(X_train_in_folds)
    auc_test_array = np.zeros((n_folds,))
    auc_train_array = np.zeros((n_folds,))
    time_of_algorithm_test = np.zeros((n_folds,))
    time_of_algorithm_train = np.zeros((n_folds,))
    for fold_index in range(n_folds):
        X_train = X_train_in_folds[fold_index]
        X_test = X_test_in_folds[fold_index]
        y_train = y_train_in_folds[fold_index]
        y_test = y_test_in_folds[fold_index]
        if fold_index == 0:
            y = list(y_train)
            y.extend(y_test)
            y = np.asarray(y)
            # print(y)
            percentage_of_anomalies = sum(y == -1) / len(y)
            print("percentage of the anomalies = " + str(percentage_of_anomalies))
        if anomaly_method == "iso_forest":
            clf = IsolationForest(random_state=rng)
            start = time.time()
            clf.fit(X=X_train)
            scores_train = clf.decision_function(X=X_train)
            end = time.time()
            time_of_algorithm_train[fold_index] = end - start
            start = time.time()
            scores_test = clf.decision_function(X=X_test)
            end = time.time()
            time_of_algorithm_test[fold_index] = end - start
        elif anomaly_method == "one_class_SVM":
            clf = OneClassSVM(gamma='auto')
            start = time.time()
            clf.fit(X=X_train)
            scores_train = clf.decision_function(X=X_train)
            end = time.time()
            time_of_algorithm_train[fold_index] = end - start
            start = time.time()
            scores_test = clf.decision_function(X=X_test)
            end = time.time()
            time_of_algorithm_test[fold_index] = end - start
        elif anomaly_method == "LOF":
            n_neighbors = 10
            clf = LOF(n_neighbors=n_neighbors, contamination=0.1)
            start = time.time()
            clf.fit(X=X_train)
            scores_train = clf.negative_outlier_factor_
            end = time.time()
            time_of_algorithm_train[fold_index] = end - start
            clf = LOF(n_neighbors=n_neighbors, novelty=True, contamination=0.1)
            start = time.time()
            clf.fit(X=X_train)
            scores_test = clf.decision_function(X=X_test)
            end = time.time()
            time_of_algorithm_test[fold_index] = end - start
        elif anomaly_method == "covariance_estimator":
            clf = EllipticEnvelope(random_state=rng)
            start = time.time()
            clf.fit(X=X_train)
            scores_train = clf.decision_function(X=X_train)
            end = time.time()
            time_of_algorithm_train[fold_index] = end - start
            start = time.time()
            scores_test = clf.decision_function(X=X_test)
            end = time.time()
            time_of_algorithm_test[fold_index] = end - start
        elif anomaly_method == "iMondrian_forest":
            settings, data, param, cache, train_ids_current_minibatch = MondrianForest.prepare_training_data(X=X_train, num_trees=100)
            clf = MondrianForest(settings, data)
            subsampling_size = 256
            start = time.time()
            # clf.fit(data, train_ids_current_minibatch, settings, param, cache, subsampling_size=None)
            clf.fit(data, train_ids_current_minibatch, settings, param, cache, subsampling_size=subsampling_size)
            scores, scores_shifted = clf.get_anomaly_scores(test_data=X_train, settings=settings, subsampling_size=None)
            scores_train = scores_shifted
            end = time.time()
            time_of_algorithm_train[fold_index] = end - start
            start = time.time()
            scores, scores_shifted = clf.get_anomaly_scores(test_data=X_test, settings=settings, subsampling_size=None)
            scores_test = scores_shifted
            end = time.time()
            time_of_algorithm_test[fold_index] = end - start
        # scores_test = -1 * scores_test  #--> to have: the more score, the less anomaly
        fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, scores_test, pos_label=1) #--> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, scores_train, pos_label=1)
        # plt.plot(fpr_test, tpr_test)
        # plt.show()
        # plt.plot(fpr_train, tpr_train)
        # plt.show()
        auc_test = metrics.auc(fpr_test, tpr_test)  #--> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
        print("Fold: " + str(fold_index) + " ---> AUC for test: " + str(auc_test))
        auc_test_array[fold_index] = auc_test
        auc_train = metrics.auc(fpr_train, tpr_train)
        print("Fold: " + str(fold_index) + " ---> AUC for train: " + str(auc_train))
        auc_train_array[fold_index] = auc_train
    auc_test_mean = auc_test_array.mean()
    auc_test_std = auc_test_array.std()
    auc_train_mean = auc_train_array.mean()
    auc_train_std = auc_train_array.std()
    time_of_algorithm_train_mean = time_of_algorithm_train.mean()
    time_of_algorithm_train_std = time_of_algorithm_train.std()
    time_of_algorithm_test_mean = time_of_algorithm_test.mean()
    time_of_algorithm_test_std = time_of_algorithm_test.std()
    print("Average AUC for test data: " + str(auc_test_mean) + " +- " + str(auc_test_std))
    print("Average time for test data: " + str(time_of_algorithm_test_mean) + " +- " + str(time_of_algorithm_test_std))
    print("Average AUC for train data: " + str(auc_train_mean) + " +- " + str(auc_train_std))
    print("Average time for train data: " + str(time_of_algorithm_train_mean) + " +- " + str(time_of_algorithm_train_std))
    if anomaly_method == "LOF" or anomaly_method == "CAD":
        path = './output/batch/' + dataset + "/" + anomaly_method + "/neigh=" + str(n_neighbors) + "/"
    else:
        path = './output/batch/' + dataset + "/" + anomaly_method + "/"
    save_np_array_to_txt(variable=auc_test_array, name_of_variable="auc_test_array", path_to_save=path)
    save_np_array_to_txt(variable=auc_test_mean, name_of_variable="auc_test_mean", path_to_save=path)
    save_np_array_to_txt(variable=auc_test_std, name_of_variable="auc_test_std", path_to_save=path)
    save_np_array_to_txt(variable=auc_train_array, name_of_variable="auc_train_array", path_to_save=path)
    save_np_array_to_txt(variable=auc_train_mean, name_of_variable="auc_train_mean", path_to_save=path)
    save_np_array_to_txt(variable=auc_train_std, name_of_variable="auc_train_std", path_to_save=path)
    save_np_array_to_txt(variable=time_of_algorithm_test, name_of_variable="time_of_algorithm_test", path_to_save=path)
    save_np_array_to_txt(variable=time_of_algorithm_test_mean, name_of_variable="time_of_algorithm_test_mean", path_to_save=path)
    save_np_array_to_txt(variable=time_of_algorithm_test_std, name_of_variable="time_of_algorithm_test_std", path_to_save=path)
    save_np_array_to_txt(variable=time_of_algorithm_train, name_of_variable="time_of_algorithm_train", path_to_save=path)
    save_np_array_to_txt(variable=time_of_algorithm_train_mean, name_of_variable="time_of_algorithm_train_mean", path_to_save=path)
    save_np_array_to_txt(variable=time_of_algorithm_train_std, name_of_variable="time_of_algorithm_train_std", path_to_save=path)
    save_np_array_to_txt(variable=percentage_of_anomalies, name_of_variable="percentage_of_anomalies", path_to_save=path)

def anomaly_detection_AUC_experiment_online(anomaly_method, dataset, X_stages, y_stages):
    rng = np.random.RandomState(42)
    n_stages = len(X_stages)
    auc_stages_array = np.zeros((n_stages,))
    time_of_algorithm_stages = np.zeros((n_stages,))
    # --- fit using the first stage of data:
    if anomaly_method == "iMondrian_forest":
        settings, data, param, cache, train_ids_current_minibatch = MondrianForest.prepare_training_data(X=X_stages[0], num_trees=100)
        clf = MondrianForest(settings, data)
        subsampling_size = 256
        start = time.time()
        clf.fit(data, train_ids_current_minibatch, settings, param, cache, subsampling_size=None)
        # clf.fit(data, train_ids_current_minibatch, settings, param, cache, subsampling_size=subsampling_size)
        end = time.time()
        time_of_algorithm_train = end - start
    elif anomaly_method == "osPCA_powerMethod":
        clf = My_osPCA(r=0.1)
        start = time.time()
        scores_stage = clf.osPCA_powerMethod_fit_first_batch(X=X_stages[0].T)
        scores_stage = scores_stage * -1
        end = time.time()
        time_of_algorithm_train = end - start
    elif anomaly_method == "osPCA_leastSquares":
        clf = My_osPCA(r=0.1)
        start = time.time()
        scores_stage = clf.osPCA_leastSquares_fit_first_batch(X=X_stages[0].T)
        scores_stage = scores_stage * -1
        end = time.time()
        time_of_algorithm_train = end - start
    elif anomaly_method == "online_LOF":
        n_neighbors = 10
        clf = My_online_LOF(n_neighbors=n_neighbors, contamination=0.1)
        start = time.time()
        clf.fit(X=X_stages[0])
        scores_stage = clf.negative_outlier_factor_
        end = time.time()
        time_of_algorithm_train = end - start
    print("Training (at first stage): ---> Time: " + str(time_of_algorithm_train))
    dimensionality_of_data = X_stages[0].shape[1]
    X_all_in_this_stage = np.empty((0, dimensionality_of_data))
    y_all_in_this_stage = []
    percentage_of_anomalies_array = np.zeros((n_stages,))
    for stage_index in range(n_stages):
        X_stage = X_stages[stage_index]
        y_stage = y_stages[stage_index]
        y = list(y_stage)
        y = np.asarray(y)
        percentage_of_anomalies = sum(y == -1) / len(y)
        print("percentage of the anomalies in this stage = " + str(percentage_of_anomalies))
        percentage_of_anomalies_array[stage_index] = percentage_of_anomalies
        if anomaly_method == "iMondrian_forest":
            start = time.time()
            if stage_index != 0:
                data, train_ids_current_minibatch = MondrianForest.prepare_new_training_data(X_train=X_stages[0], X_new=X_stage)
                clf.partial_fit(data=data, train_ids_current_minibatch=train_ids_current_minibatch, settings=settings, param=param, cache=cache)
            X_all_in_this_stage = np.vstack((X_all_in_this_stage, X_stages[stage_index]))
            scores, scores_shifted = clf.get_anomaly_scores(test_data=X_all_in_this_stage, settings=settings, subsampling_size=None)
            scores_stage = scores_shifted
            end = time.time()
            time_of_algorithm_stages[stage_index] = end - start
        elif anomaly_method == "osPCA_powerMethod":
            if stage_index == 0:
                time_of_algorithm_stages[stage_index] = 0
                scores_stage = scores_stage
            else:
                start = time.time()
                scores = clf.osPCA_powerMethod_fit_new_batch(X_new=X_stages[stage_index].T)
                scores = scores * -1
                temp = list(scores_stage)
                temp.extend(list(scores))
                scores_stage = np.asarray(temp)
                end = time.time()
                time_of_algorithm_stages[stage_index] = end - start
        elif anomaly_method == "osPCA_leastSquares":
            if stage_index == 0:
                time_of_algorithm_stages[stage_index] = 0
                scores_stage = scores_stage
            else:
                start = time.time()
                scores = clf.osPCA_leastSquares_fit_new_batch(X_new=X_stages[stage_index].T)
                scores = scores * -1
                temp = list(scores_stage)
                temp.extend(list(scores))
                scores_stage = np.asarray(temp)
                end = time.time()
                time_of_algorithm_stages[stage_index] = end - start
        elif anomaly_method == "online_LOF":
            if stage_index == 0:
                time_of_algorithm_stages[stage_index] = 0
                scores_stage = scores_stage
            else:
                start = time.time()
                clf.fit_new_data(X_newData=X_stages[stage_index])
                scores = clf.negative_outlier_factor_
                scores_stage = scores
                end = time.time()
                time_of_algorithm_stages[stage_index] = end - start
        if stage_index == 0:
            time_of_algorithm_stages[stage_index] = time_of_algorithm_stages[stage_index] + time_of_algorithm_train
        # scores_test = -1 * scores_test  #--> to have: the more score, the less anomaly
        y_all_in_this_stage.extend(y_stage)
        fpr_stage, tpr_stage, thresholds_stage = metrics.roc_curve(y_all_in_this_stage, scores_stage, pos_label=1)
        # plt.plot(fpr_train, tpr_train)
        # plt.show()
        auc_stage = metrics.auc(fpr_stage, tpr_stage)
        print("Stage: " + str(stage_index) + " ---> AUC for this stage: " + str(auc_stage))
        print("Stage: " + str(stage_index) + " ---> Time for this stage: " + str(time_of_algorithm_stages[stage_index]))
        auc_stages_array[stage_index] = auc_stage
    auc_stages_mean = auc_stages_array.mean()
    auc_stages_std = auc_stages_array.std()
    time_stages_mean = time_of_algorithm_stages.mean()
    time_stages_std = time_of_algorithm_stages.std()
    print("Average AUC for stages: " + str(auc_stages_mean) + " +- " + str(auc_stages_std))
    print("Average time for stages: " + str(time_stages_mean) + " +- " + str(time_stages_std))
    if anomaly_method == "LOF" or anomaly_method == "CAD":
        pass
        # path = './output/online/' + dataset + "/" + anomaly_method + "/neigh=" + str(n_neighbors) + "/"
    else:
        path = './output/online/' + dataset + "/" + anomaly_method + "/"
    save_np_array_to_txt(variable=auc_stages_array, name_of_variable="auc_stages_array", path_to_save=path)
    save_np_array_to_txt(variable=auc_stages_mean, name_of_variable="auc_stages_mean", path_to_save=path)
    save_np_array_to_txt(variable=auc_stages_std, name_of_variable="auc_stages_std", path_to_save=path)
    save_np_array_to_txt(variable=np.asarray(time_of_algorithm_train), name_of_variable="time_of_algorithm_train", path_to_save=path)
    save_np_array_to_txt(variable=time_of_algorithm_stages, name_of_variable="time_of_algorithm_stages", path_to_save=path)
    save_np_array_to_txt(variable=time_stages_mean, name_of_variable="time_stages_mean", path_to_save=path)
    save_np_array_to_txt(variable=time_stages_std, name_of_variable="time_stages_std", path_to_save=path)
    save_np_array_to_txt(variable=percentage_of_anomalies_array, name_of_variable="percentage_of_anomalies_array", path_to_save=path)

def anomaly_visual_experiment_bach(dataset, method, generate_synthetic_datasets_again, predict_anomaly_again, predict_using_threshold):
    # https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html#sphx-glr-auto-examples-plot-anomaly-comparison-py
    # settings:
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers
    xx, yy = np.meshgrid(np.linspace(-7, 7, 150), np.linspace(-7, 7, 150))
    rng = np.random.RandomState(42)
    # dataset:
    path_dataset = './datasets/' + dataset + "/"
    if generate_synthetic_datasets_again:
        if dataset == "two_moons":
            X = 4. * (make_moons(n_samples=n_inliers, noise=.05, random_state=0)[0] - np.array([0.5, 0.25]))
        elif dataset == "one_blob":
            X = make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5)[0]
        elif dataset == "two_blobs":
            X = make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5])[0]
        elif dataset == "two_different_blobs":
            X = make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3])[0]
        save_variable(variable=X, name_of_variable="X", path_to_save=path_dataset)
    else:
        X = load_variable(name_of_variable="X", path=path_dataset)
    # Add outliers:
    X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
    # anomaly detection algorithm:
    if method == "iMondrian_forest":
        if predict_using_threshold:
            path_save = "./saved_files/batch/" + method + "/threshold/" + dataset + "/"
        else:
            path_save = "./saved_files/batch/" + method + "/Kmeans/" + dataset + "/"
    else:
        path_save = "./saved_files/batch/" + method + "/" + dataset + "/"
    if predict_anomaly_again:
        if method == "iso_forest":
            clf = IsolationForest(contamination=outliers_fraction, random_state=42, behaviour='old')
            clf.fit(X)
            y_pred = clf.predict(X)
            print(y_pred)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            scores = clf.decision_function(X=np.c_[xx.ravel(), yy.ravel()])
            scores = scores.reshape(xx.shape)
            scores = scores - 0.5  #--> adding back the self.offset_ which is -0.5
            scores = scores * -1  #--> multipyling by -1 again
        elif method == "iMondrian_forest":
            settings, data, param, cache, train_ids_current_minibatch = MondrianForest.prepare_training_data(X=X, num_trees=100)
            mf = MondrianForest(settings, data)
            subsampling_size = 256
            mf.fit(data, train_ids_current_minibatch, settings, param, cache, subsampling_size=None)
            scores_training, _ = mf.get_anomaly_scores(test_data=X, settings=settings, subsampling_size=None)
            if predict_using_threshold:
                y_pred = mf.predict_using_threshold(anomaly_scores=scores_training, threshold=0.5)
            else:
                y_pred, which_cluster_is_anomaly, kmeans = mf.predict_using_kmeans(anomaly_scores=scores_training)
            scores, _ = mf.get_anomaly_scores(test_data=np.c_[xx.ravel(), yy.ravel()], settings=settings, subsampling_size=None)
            if predict_using_threshold:
                Z = mf.predict_using_threshold(anomaly_scores=scores, threshold=0.5)
            else:
                Z = mf.predict_outOfSample_using_kmeans(anomaly_scores=scores, which_cluster_is_anomaly=which_cluster_is_anomaly, kmeans=kmeans)
            Z = Z.reshape(xx.shape)
            scores = scores.reshape(xx.shape)
        save_variable(variable=y_pred, name_of_variable="y_pred", path_to_save=path_save)
        save_variable(variable=Z, name_of_variable="Z", path_to_save=path_save)
        save_variable(variable=scores, name_of_variable="scores", path_to_save=path_save)
    else:
        y_pred = load_variable(name_of_variable="y_pred", path=path_save)
        Z = load_variable(name_of_variable="Z", path=path_save)
        scores = load_variable(name_of_variable="scores", path=path_save)
    # ------ legends:
    # # colors = np.array(['#377eb8', '#ff7f00']) #--> https://htmlcolorcodes.com/
    # colors = np.array(['#BBFF33', '#ff7f00'])
    # markers = np.array(['^', 'o'])
    # plt.scatter(0, 0, color=colors[1], marker=markers[1], edgecolors="k")
    # plt.scatter(1, 1, color=colors[0], marker=markers[0], edgecolors="k")
    # plt.legend(["normal", "anomaly"])
    # plt.show()
    # ------ plot the predicted anomaly for the space:
    # plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    # plt.imshow(scores, cmap='hot', interpolation='nearest')
    plt.imshow(Z * -1, cmap='gray', alpha=0.2)
    # plt.colorbar()
    colors = np.array(['#BBFF33', '#ff7f00'])
    markers = np.array(['^', 'o'])
    # plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2], marker="o")
    colors_vector = colors[(y_pred + 1) // 2]
    markers_vector = markers[(y_pred + 1) // 2]
    for _s, c, _x, _y in zip(markers_vector, colors_vector, X[:, 0], X[:, 1]):
        _x = (_x + 7) * (150 / 14)
        _y = (_y + 7) * (150 / 14)
        plt.scatter(_x, _y, marker=_s, c=c, alpha=1, edgecolors="k")
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    # plt.xlim(-7, 7)
    # plt.ylim(-7, 7)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    # ------ plot the anomaly score for the space:
    plt.imshow(scores, cmap='gray')
    # plt.clim(0, 1)
    plt.colorbar()
    # plt.xlim(-7, 7)
    # plt.ylim(-7, 7)
    # plt.show()
    colors = np.array(['#BBFF33', '#ff7f00'])
    markers = np.array(['^', 'o'])
    # plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2], marker="o")
    colors_vector = colors[(y_pred + 1) // 2]
    markers_vector = markers[(y_pred + 1) // 2]
    for _s, c, _x, _y in zip(markers_vector, colors_vector, X[:, 0], X[:, 1]):
        _x = (_x + 7) * (150 / 14)
        _y = (_y + 7) * (150 / 14)
        plt.scatter(_x, _y, marker=_s, c=c, alpha=0.5, edgecolors="k")
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    # plt.xlim(-7, 7)
    # plt.ylim(-7, 7)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def anomaly_visual_experiment_online(dataset, method, generate_synthetic_datasets_again, predict_anomaly_again, predict_using_threshold):
    # https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html#sphx-glr-auto-examples-plot-anomaly-comparison-py
    # settings:
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers
    xx, yy = np.meshgrid(np.linspace(-7, 7, 150), np.linspace(-7, 7, 150))
    rng = np.random.RandomState(42)
    # dataset:
    path_dataset = './datasets/' + dataset + "/"
    if generate_synthetic_datasets_again:
        if dataset == "two_moons":
            X = 4. * (make_moons(n_samples=n_inliers, noise=.05, random_state=0)[0] - np.array([0.5, 0.25]))
        elif dataset == "one_blob":
            X = make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5)[0]
            n_inliers = X.shape[0]
        elif dataset == "two_blobs":
            X = make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5])[0]
            n_inliers = X.shape[0]
        elif dataset == "two_different_blobs":
            X = make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3])[0]
            n_inliers = X.shape[0]
        save_variable(variable=X, name_of_variable="X", path_to_save=path_dataset)
    else:
        X = load_variable(name_of_variable="X", path=path_dataset)
        n_inliers = X.shape[0]
    # Add outliers:
    X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
    y = [1] * n_inliers
    y.extend([-1] * n_outliers)
    y = np.asarray(y).ravel()
    # split dataset for online stages:
    n_online_stages = 5
    X_stages, y_stages = stratified_split_of_data(X, y, n_online_stages=n_online_stages)
    # anomaly detection:
    if method == "iMondrian_forest":
        if predict_using_threshold:
            path_save = "./saved_files/online/" + method + "/threshold/" + dataset + "/"
        else:
            path_save = "./saved_files/online/" + method + "/Kmeans/" + dataset + "/"
    else:
        path_save = "./saved_files/online/" + method + "/" + dataset + "/"
    if predict_anomaly_again:
        if method == "iMondrian_forest":
            settings, data, param, cache, train_ids_current_minibatch = MondrianForest.prepare_training_data(X=X_stages[0], num_trees=100)
            mf = MondrianForest(settings, data)
            subsampling_size = 256
            mf.fit(data, train_ids_current_minibatch, settings, param, cache, subsampling_size=None)
            y_pred_of_stages = [None] * n_online_stages
            scores_of_stages = [None] * n_online_stages
            Z_of_stages = [None] * n_online_stages
            X_all_in_this_stage = np.empty((0, X.shape[1]))
            X_in_stages = [None] * n_online_stages
            for stage in range(n_online_stages):
                print("Online stage " + str(stage) + "...")
                if stage != 0:
                    data, train_ids_current_minibatch = MondrianForest.prepare_new_training_data(X_train=X_stages[0], X_new=X_stages[stage])
                    mf.partial_fit(data=data, train_ids_current_minibatch=train_ids_current_minibatch, settings=settings, param=param, cache=cache)
                X_all_in_this_stage = np.vstack((X_all_in_this_stage, X_stages[stage]))
                scores_training, _ = mf.get_anomaly_scores(test_data=X_all_in_this_stage, settings=settings, subsampling_size=None)
                if predict_using_threshold:
                    y_pred_of_stages[stage] = mf.predict_using_threshold(anomaly_scores=scores_training, threshold=0.5)
                else:
                    y_pred_of_stages[stage], which_cluster_is_anomaly, kmeans = mf.predict_using_kmeans(anomaly_scores=scores_training)
                scores, _ = mf.get_anomaly_scores(test_data=np.c_[xx.ravel(), yy.ravel()], settings=settings, subsampling_size=None)
                if predict_using_threshold:
                    Z = mf.predict_using_threshold(anomaly_scores=scores, threshold=0.5)
                else:
                    Z = mf.predict_outOfSample_using_kmeans(anomaly_scores=scores, which_cluster_is_anomaly=which_cluster_is_anomaly, kmeans=kmeans)
                Z = Z.reshape(xx.shape)
                scores = scores.reshape(xx.shape)
                scores_of_stages[stage] = scores
                Z_of_stages[stage] = Z
                X_in_stages[stage] = X_all_in_this_stage
        save_variable(variable=y_pred_of_stages, name_of_variable="y_pred_of_stages", path_to_save=path_save)
        save_variable(variable=Z_of_stages, name_of_variable="Z_of_stages", path_to_save=path_save)
        save_variable(variable=scores_of_stages, name_of_variable="scores_of_stages", path_to_save=path_save)
        save_variable(variable=X_in_stages, name_of_variable="X_in_stages", path_to_save=path_save)
    else:
        y_pred_of_stages = load_variable(name_of_variable="y_pred_of_stages", path=path_save)
        Z_of_stages = load_variable(name_of_variable="Z_of_stages", path=path_save)
        scores_of_stages = load_variable(name_of_variable="scores_of_stages", path=path_save)
        X_in_stages = load_variable(name_of_variable="X_in_stages", path=path_save)
    # ------ legends:
    # # colors = np.array(['#377eb8', '#ff7f00']) #--> https://htmlcolorcodes.com/
    # colors = np.array(['#BBFF33', '#ff7f00'])
    # markers = np.array(['^', 'o'])
    # plt.scatter(0, 0, color=colors[1], marker=markers[1], edgecolors="k")
    # plt.scatter(1, 1, color=colors[0], marker=markers[0], edgecolors="k")
    # plt.legend(["normal", "anomaly"])
    # plt.show()
    for stage in range(n_online_stages):
        y_pred = y_pred_of_stages[stage]
        scores = scores_of_stages[stage]
        Z = Z_of_stages[stage]
        X = X_in_stages[stage]
        # ------ plot the predicted anomaly for the space:
        # plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        # plt.imshow(scores, cmap='hot', interpolation='nearest')
        plt.imshow(Z * -1, cmap='gray', alpha=0.2)
        # plt.colorbar()
        colors = np.array(['#BBFF33', '#ff7f00'])
        markers = np.array(['^', 'o'])
        # plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2], marker="o")
        colors_vector = colors[(y_pred + 1) // 2]
        markers_vector = markers[(y_pred + 1) // 2]
        for _s, c, _x, _y in zip(markers_vector, colors_vector, X[:, 0], X[:, 1]):
            _x = (_x + 7) * (150 / 14)
            _y = (_y + 7) * (150 / 14)
            plt.scatter(_x, _y, marker=_s, c=c, alpha=1, edgecolors="k")
        plt.xlim(0, 150)
        plt.ylim(0, 150)
        # plt.xlim(-7, 7)
        # plt.ylim(-7, 7)
        plt.xticks(())
        plt.yticks(())
        plt.show()
        # ------ plot the anomaly score for the space:
        plt.imshow(scores, cmap='gray')
        if stage == 0:
            a = np.min(scores)
            b = np.max(scores)
        plt.clim(a, b)
        plt.colorbar()
        # plt.xlim(-7, 7)
        # plt.ylim(-7, 7)
        # plt.show()
        colors = np.array(['#BBFF33', '#ff7f00'])
        markers = np.array(['^', 'o'])
        # plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2], marker="o")
        colors_vector = colors[(y_pred + 1) // 2]
        markers_vector = markers[(y_pred + 1) // 2]
        for _s, c, _x, _y in zip(markers_vector, colors_vector, X[:, 0], X[:, 1]):
            _x = (_x + 7) * (150 / 14)
            _y = (_y + 7) * (150 / 14)
            plt.scatter(_x, _y, marker=_s, c=c, alpha=0.5, edgecolors="k")
        plt.xlim(0, 150)
        plt.ylim(0, 150)
        # plt.xlim(-7, 7)
        # plt.ylim(-7, 7)
        plt.xticks(())
        plt.yticks(())
        plt.show()


def stratified_split_of_data(X, y, n_online_stages):
    # X: rows are samples and columns are features
    size_of_every_stage = int(X.shape[0] / n_online_stages)
    sss = StratifiedShuffleSplit(n_splits=1, random_state=10, test_size=size_of_every_stage)
    X_new = X
    y_new = y
    X_stages = [None] * n_online_stages
    y_stages = [None] * n_online_stages
    for stage in range(n_online_stages-1):
        print("Splitting data: stage " + str(stage))
        for train_index, test_index in sss.split(X_new, y_new):
            X_train, X_test = X_new[train_index, :], X_new[test_index, :]
            y_train, y_test = y_new[train_index], y_new[test_index]
            X_stages[stage] = X_test
            X_new = X_train
            y_stages[stage] = y_test
            y_new = y_train
    X_stages[-1] = X_new
    y_stages[-1] = y_new
    return X_stages, y_stages


def experiment_original_code():
    # DATASET_FILE = '/Users/gary/Downloads/cardio.mat'
    DATASET_FILE = './testframe/thyroid.mat'
    X, y = load_mat(DATASET_FILE)
    # DATASET_FILE = '/home/gary/school/imondrian-forests/algorithms_for_comparison/{x}/{x}_imondrian_tr.csv'.format(
    #     x='abalone',
    # )
    # X, y = load_csv(DATASET_FILE)
    y_backup = y.copy()
    y = np.ones((X.shape[0],))
    print(X.shape)
    print(y.shape)
    aucs = []
    N = 5
    for _ in range(N):
        mf, settings, data = train_for_experiment_original_code(X, y, num_trees=100)
        aucs.append(evaluate_for_experiment_original_code(y_backup[:100], mf.get_anomaly_scores(X[:100, ], settings)))
    print(f'mean auc over {N} runs = ', np.mean(np.array(aucs)))
    # demo()

def format_data_dict(X, y):
    return {
        'is_sparse': False,
        'n_class': 2,
        'n_dim': len(X[0]),
        'n_train': len(X),
        'n_test': 0,
        'train_ids_partition': {
            'cumulative': {0: np.array(list(range(len(X))))},
            'current': {0: np.array(list(range(len(X))))}
        },
        'x_train': X,
        'y_train': y,
        'x_test': np.array([]),
        'y_test': np.array([]),
    }

def load_mat(path):
    """
    load matlab files
    format: X = Multi-dimensional point data, y = labels (1 = outliers, 0 = inliers)
    """
    data = loadmat(path)
    return data['X'].astype(float), data['y'].flatten().astype(int)

def load_csv(path, skip=1):
    data = np.genfromtxt(path, delimiter=',', dtype=None)[skip:]
    return data[:, :-1].astype(float), (data[:, -1] == b'"anomaly"').astype(int)

def train_for_experiment_original_code(X, y, num_trees=100, subsampling_size=256):
    argv = [
        '--n_minibatches', '1',
        '--n_mondrians', str(num_trees),
        '--budget', '-1',
        '--normalize_features', '1',
        '--optype', 'real',  #--> 'class' or 'real'
        '--draw_mondrian', '0',
        '--isolate',
    ]
    settings = process_command_line(argv)
    reset_random_seed(settings)
    data = format_data_dict(X, y)
    param, cache = precompute_minimal(data, settings)
    mf = MondrianForest(settings, data)
    print(('Training on %d samples of dimension %d' % (data['n_train'], data['n_dim'])))
    train_ids_current_minibatch = data['train_ids_partition']['current'][0]
    mf.fit(data, train_ids_current_minibatch, settings, param, cache, subsampling_size=subsampling_size)
    print('\nFinal forest stats:')
    tree_stats = np.zeros((settings.n_mondrians, 2))
    tree_average_depth = np.zeros(settings.n_mondrians)
    for i_t, tree in enumerate(mf.forest):
        tree_stats[
        i_t, -
             2:
        ] = np.array([len(tree.leaf_nodes), len(tree.non_leaf_nodes)])
        tree_average_depth[i_t] = tree.get_average_depth(settings, data)[0]
    print(('mean(num_leaves) = {:.1f}, mean(num_non_leaves) = {:.1f}, mean(tree_average_depth) = {:.1f}'.format(
        np.mean(tree_stats[:, -2]), np.mean(tree_stats[:, -1]), np.mean(tree_average_depth))))
    print(('n_train = %d, log_2(n_train) = %.1f, mean(tree_average_depth) = %.1f +- %.1f' %
           (data['n_train'], np.log2(data['n_train']), np.mean(tree_average_depth), np.std(tree_average_depth))))
    return mf, settings, data


def evaluate_for_experiment_original_code(y_true, y_score):
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    print('N samples = ', len(y_true))
    print('% true outlier = ', np.count_nonzero(y_true) / len(y_true))
    print('auc =', roc_auc)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return roc_auc

def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def load_variable(name_of_variable, path='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable

def save_np_array_to_txt(variable, name_of_variable, path_to_save='./'):
    if type(variable) is list:
        variable = np.asarray(variable)
    # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.txt'
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    with open(file_address, 'w') as f:
        f.write(np.array2string(variable, separator=', '))

def read_BreastCancer_dataset(data, labels_of_classes):
    data = data.values  # converting pandas dataFrame to numpy array
    labels = data[:,1]
    total_number_of_samples = data.shape[0]
    X = data[:,2:]
    X = X.astype(np.float32)  # if we don't do that, we will have this error: https://www.reddit.com/r/learnpython/comments/7ivopz/numpy_getting_error_on_matrix_inverse/
    y = [None] * (total_number_of_samples)  # numeric labels
    for sample_index in range(total_number_of_samples):
        if labels[sample_index] == labels_of_classes[0]:  # first class --> M
            y[sample_index] = 0
        elif labels[sample_index] == labels_of_classes[1]:  # second class --> B
            y[sample_index] = 1
    return X, y

def cross_validation(X, y, n_splits=10):
    # sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=None)
    CV = KFold(n_splits=n_splits, random_state=100, shuffle=True)
    train_indices_in_folds = []; test_indices_in_folds = []
    X_train_in_folds = []; X_test_in_folds = []
    y_train_in_folds = []; y_test_in_folds = []
    for train_index, test_index in CV.split(X, y):
        train_indices_in_folds.append(train_index)
        test_indices_in_folds.append(test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np.asarray(y)[train_index], np.asarray(y)[test_index]
        X_train_in_folds.append(X_train)
        X_test_in_folds.append(X_test)
        y_train_in_folds.append(y_train)
        y_test_in_folds.append(y_test)
    return train_indices_in_folds, test_indices_in_folds, X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds

if __name__ == '__main__':
    main()
