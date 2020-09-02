import numpy as np
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import KMeans
import time


class My_osPCA:

    def __init__(self, r):
        # X: rows are features and columns are samples
        self.u_so_far = None
        self.r = r  #--> r \in (0, 1): oversampling ratio; r = n_oversampling / n_total
        self.X_so_far = None

    def osPCA_leastSquares_fit_first_batch(self, X):
        # X: first batch --> rows are features, columns are samples
        n_samples = X.shape[1]
        n_dimensions = X.shape[0]
        scores = np.zeros((n_samples,))
        for sample_index in range(n_samples):
            sample = X[:, sample_index].reshape((-1, 1))
            X_without_a_sample = np.delete(arr=X, obj=sample_index, axis=1)
            _, U = self.calculate_SVD(X=X_without_a_sample)
            u = U[:, 0].reshape((-1, 1))
            mean_so_far = X_without_a_sample.mean(axis=1).reshape((-1, 1))
            n_samples_so_far = X_without_a_sample.shape[1]
            x_t_bar = sample - mean_so_far
            y_t = (u.T).dot(x_t_bar)
            temp1, temp2 = 0, 0
            for sample_index in range(n_samples_so_far):
                sample = X_without_a_sample[:, sample_index].reshape((-1, 1))
                x_i_bar = sample - mean_so_far
                y_i = (u.T).dot(x_i_bar)
                temp1 = temp1 + (y_i * x_i_bar)
                temp2 = temp2 + (y_i ** 2)
            beta = 1 / (n_samples_so_far * self.r)
            numerator = (beta * temp1) + (y_t * x_t_bar)
            denominator = (beta * temp2) + (y_t ** 2)
            u_new = numerator / denominator
            scores[sample_index] = self.calculate_anomaly_score(u=u, u_new=u_new)
        self.X_so_far = np.empty((n_dimensions, 0))
        self.update_X_and_u_so_far(X_new=X)
        return scores

    def osPCA_leastSquares_calculate_new_u(self, x_new):
        # x_new: a column vector
        mean_so_far = self.X_so_far.mean(axis=1).reshape((-1, 1))
        n_samples_so_far = self.X_so_far.shape[1]
        x_t_bar = x_new - mean_so_far
        y_t = (self.u_so_far.T).dot(x_t_bar)
        temp1, temp2 = 0, 0
        for sample_index in range(n_samples_so_far):
            sample = self.X_so_far[:, sample_index].reshape((-1, 1))
            x_i_bar = sample - mean_so_far
            y_i = (self.u_so_far.T).dot(x_i_bar)
            temp1 = temp1 + (y_i * x_i_bar)
            temp2 = temp2 + (y_i ** 2)
        beta = 1 / (n_samples_so_far * self.r)
        numerator = (beta * temp1) + (y_t * x_t_bar)
        denominator = (beta * temp2) + (y_t ** 2)
        u_new = numerator / denominator
        self.update_X_and_u_so_far(X_new=x_new)
        return u_new

    def osPCA_leastSquares_fit_new_vector(self, x_new):
        # x_new: a column vector
        u_new = self.osPCA_leastSquares_calculate_new_u(x_new=x_new)
        score = self.calculate_anomaly_score(u=self.u_so_far, u_new=u_new)
        self.update_X_and_u_so_far(X_new=x_new)
        return score

    def osPCA_leastSquares_fit_new_batch(self, X_new):
        # X_new: new batch --> rows are features, columns are samples
        # notice: the order of new samples in X_new matters
        n_new_samples = X_new.shape[1]
        scores = np.zeros((n_new_samples,))
        for sample_index in range(n_new_samples):
            new_sample = X_new[:, sample_index].reshape((-1, 1))
            scores[sample_index] = self.osPCA_leastSquares_fit_new_vector(x_new=new_sample)
        return scores

    def osPCA_powerMethod_fit_first_batch(self, X):
        # X: first batch --> rows are features, columns are samples
        n_samples = X.shape[1]
        n_dimensions = X.shape[0]
        scores = np.zeros((n_samples,))
        for sample_index in range(n_samples):
            sample = X[:, sample_index]
            X_without_a_sample = np.delete(arr=X, obj=sample_index, axis=1)
            covariance_matrix = self.covariance_matrix(X=X_without_a_sample)
            u = self.power_method(matrix=covariance_matrix)
            new_covariance = self.covariance_matrix_withOversampledNewData(X=X_without_a_sample, x_new=sample)
            u_new = self.power_method(matrix=new_covariance)
            scores[sample_index] = self.calculate_anomaly_score(u=u, u_new=u_new)
        self.X_so_far = np.empty((n_dimensions, 0))
        self.update_X_and_u_so_far(X_new=X)
        return scores

    def update_X_and_u_so_far(self, X_new):
        # X_new: a column vector or a column-wise matrix
        self.X_so_far = np.column_stack((self.X_so_far, X_new))
        covariance_matrix = self.covariance_matrix(X=self.X_so_far)
        self.u_so_far = self.power_method(matrix=covariance_matrix)

    def osPCA_powerMethod_fit_new_vector(self, x_new):
        # x_new: a column vector
        new_covariance = self.covariance_matrix_withOversampledNewData(X=self.X_so_far, x_new=x_new)
        u_new = self.power_method(matrix=new_covariance)
        score = self.calculate_anomaly_score(u=self.u_so_far, u_new=u_new)
        self.update_X_and_u_so_far(X_new=x_new)
        return score

    def osPCA_powerMethod_fit_new_batch(self, X_new):
        # X_new: new batch --> rows are features, columns are samples
        # notice: the order of new samples in X_new matters
        n_new_samples = X_new.shape[1]
        scores = np.zeros((n_new_samples,))
        for sample_index in range(n_new_samples):
            new_sample = X_new[:, sample_index]
            scores[sample_index] = self.osPCA_powerMethod_fit_new_vector(x_new=new_sample)
        return scores

    def calculate_anomaly_score(self, u, u_new):
        temp1 = (u.T).dot(u_new)
        temp2 = LA.norm(u) * LA.norm(u_new)
        cosine_abs = np.abs(temp1 / temp2)
        score = 1 - cosine_abs
        return score

    def covariance_matrix_withOversampledNewData(self, X, x_new):
        # X: rows are features, columns are samples, x: a column vector, r \in (0, 1): oversampling ratio; r = n_oversampling / n_total
        n_samples = X.shape[1]
        mean_of_first_batch = X.mean(axis=1).reshape((-1, 1))
        Q = (1 / n_samples) * X.dot(X.T)
        temp1 = (1 / (1 + self.r)) * Q
        temp2 = (self.r / (1 + self.r)) * x_new.dot(x_new.T)
        temp3 = mean_of_first_batch.dot(mean_of_first_batch.T)
        new_covariance = temp1 + temp2 - temp3
        return new_covariance

    def power_method(self, matrix):
        n_dimensions = matrix.shape[0]
        u = np.random.rand(n_dimensions, 1)
        for _ in range(1000):
            a = matrix.dot(u)
            u = a / LA.norm(a)
        return u

    def covariance_matrix(self, X):
        # X: rows are features, columns are samples
        n_samples = X.shape[1]
        mean_of_data = X.mean(axis=1).reshape((-1, 1))
        X_centered = X - np.tile(mean_of_data, n_samples)
        covariance_matrix = X_centered.dot(X_centered.T)
        covariance_matrix = (1 / n_samples) * covariance_matrix
        return covariance_matrix

    def calculate_eivenvectors(self, matrix):
        eig_val, eig_vec = LA.eigh(matrix)
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        return eig_val, eig_vec

    def calculate_SVD(self, X):
        U, s, Vh = LA.svd(X, full_matrices=False)  # ---> in dual PCA, the S should be square so --> full_matrices=False
        V = Vh.T
        left_singular_matrix = U
        singular_values = s
        return singular_values, left_singular_matrix

    def center_the_matrix(self, the_matrix, mode="double_center"):
        n_rows = the_matrix.shape[0]
        n_cols = the_matrix.shape[1]
        vector_one_left = np.ones((n_rows, 1))
        vector_one_right = np.ones((n_cols, 1))
        H_left = np.eye(n_rows) - ((1 / n_rows) * vector_one_left.dot(vector_one_left.T))
        H_right = np.eye(n_cols) - ((1 / n_cols) * vector_one_right.dot(vector_one_right.T))
        if mode == "double_center":
            the_matrix = H_left.dot(the_matrix).dot(H_right)
        elif mode == "remove_mean_of_rows_from_rows":
            the_matrix = H_left.dot(the_matrix)
        elif mode == "remove_mean_of_columns_from_columns":
            the_matrix = the_matrix.dot(H_right)
        return the_matrix