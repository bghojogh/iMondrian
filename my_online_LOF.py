import numpy as np
import warnings

from sklearn.neighbors import NearestNeighbors


class My_online_LOF:

    def __init__(self, n_neighbors=10, algorithm='auto', metric='minkowski', p=2, contamination=0.1, novelty=False, n_jobs=None):
        self.contamination = contamination
        self.novelty = novelty
        self.n_neighbors = n_neighbors
        self.n_neighbors_ = None
        self._distances_fit_X_ = None
        self.n_jobs = n_jobs
        self._contamination = contamination
        self.offset_ = None
        self.X = None

    def fit_predict(self, X, y=None):
        """"Fits the model to the training set X and returns the labels.
        Label is 1 for an inlier and -1 for an outlier according to the LOF
        score and the contamination parameter.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features), default=None
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. to the training samples.
        y : Ignored
            not used, present for API consistency by convention.
        Returns
        -------
        is_inlier : array, shape (n_samples,)
            Returns -1 for anomalies/outliers and 1 for inliers.
        """
        self.fit(X)
        y_pred = self.predict(X)
        return y_pred

    def fit(self, X, y=None):
        """Fit the model using X as training data.
        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.
        y : Ignored
            not used, present for API consistency by convention.
        """
        self.X = X
        n_samples = X.shape[0]
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors_, algorithm='ball_tree', n_jobs=self.n_jobs).fit(X)
        self._distances_fit_X_, _neighbors_indices_fit_X_ = nbrs.kneighbors(X)
        self._lrd = self._local_reachability_density(distances_X=self._distances_fit_X_, neighbors_indices=_neighbors_indices_fit_X_)
        # Compute lof score over training samples to define offset_:
        lrd_ratios_array = (self._lrd[_neighbors_indices_fit_X_] / self._lrd[:, np.newaxis])
        self.negative_outlier_factor_ = -np.mean(lrd_ratios_array, axis=1)
        if self._contamination == "auto":
            # inliers score around -1 (the higher, the less abnormal).
            self.offset_ = -1.5
        else:
            self.offset_ = np.percentile(self.negative_outlier_factor_, 100. * self._contamination)
        return self

    def fit_new_data(self, X_newData):
        # X_newData: rows are samples and columns are features
        X = np.vstack((self.X, X_newData))
        self.X = X
        n_samples = X.shape[0]
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors_, algorithm='ball_tree', n_jobs=self.n_jobs).fit(X)
        self._distances_fit_X_, _neighbors_indices_fit_X_ = nbrs.kneighbors(X)
        self._lrd = self._local_reachability_density(distances_X=self._distances_fit_X_, neighbors_indices=_neighbors_indices_fit_X_)
        # Compute lof score over training samples to define offset_:
        lrd_ratios_array = (self._lrd[_neighbors_indices_fit_X_] / self._lrd[:, np.newaxis])
        self.negative_outlier_factor_ = -np.mean(lrd_ratios_array, axis=1)
        if self._contamination == "auto":
            # inliers score around -1 (the higher, the less abnormal).
            self.offset_ = -1.5
        else:
            self.offset_ = np.percentile(self.negative_outlier_factor_, 100. * self._contamination)
        return self

    def predict(self, X):
        """Predict the labels (1 inlier, -1 outlier) of X according to LOF.
        If X is None, returns the same as fit_predict(X_train).
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features), default=None
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. to the training samples. If None, makes prediction on the
            training data without considering them as their own neighbors.
        Returns
        -------
        is_inlier : array, shape (n_samples,)
            Returns -1 for anomalies/outliers and +1 for inliers.
        """
        y_pred = np.ones(X.shape[0], dtype=int)
        y_pred[self.decision_function(X) < 0] = -1
        return y_pred

    def decision_function(self, X):
        """Shifted opposite of the Local Outlier Factor of X.
        Bigger is better, i.e. large values correspond to inliers.
        The shift offset allows a zero threshold for being an outlier.
        Only available for novelty detection (when novelty is set to True).
        The argument X is supposed to contain *new data*: if X contains a
        point from training, it considers the later in its own neighborhood.
        Also, the samples in X are not considered in the neighborhood of any
        point.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. the training samples.
        Returns
        -------
        shifted_opposite_lof_scores : array, shape (n_samples,)
            The shifted opposite of the Local Outlier Factor of each input
            samples. The lower, the more abnormal. Negative scores represent
            outliers, positive scores represent inliers.
        """
        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        """Opposite of the Local Outlier Factor of X.
        It is the opposite as bigger is better, i.e. large values correspond
        to inliers.
        Only available for novelty detection (when novelty is set to True).
        The argument X is supposed to contain *new data*: if X contains a
        point from training, it considers the later in its own neighborhood.
        Also, the samples in X are not considered in the neighborhood of any
        point.
        The score_samples on training data is available by considering the
        the ``negative_outlier_factor_`` attribute.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. the training samples.
        Returns
        -------
        opposite_lof_scores : array, shape (n_samples,)
            The opposite of the Local Outlier Factor of each input samples.
            The lower, the more abnormal.
        """
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors_, algorithm='ball_tree', n_jobs=self.n_jobs).fit(X)
        distances_X, neighbors_indices_X = nbrs.kneighbors(X)
        X_lrd = self._local_reachability_density(distances_X, neighbors_indices_X)
        lrd_ratios_array = (self._lrd[neighbors_indices_X] / X_lrd[:, np.newaxis])
        # as bigger is better:
        return -np.mean(lrd_ratios_array, axis=1)

    def _local_reachability_density(self, distances_X, neighbors_indices):
        """The local reachability density (LRD)
        The LRD of a sample is the inverse of the average reachability
        distance of its k-nearest neighbors.
        Parameters
        ----------
        distances_X : array, shape (n_query, self.n_neighbors)
            Distances to the neighbors (in the training samples `self._fit_X`)
            of each query point to compute the LRD.
        neighbors_indices : array, shape (n_query, self.n_neighbors)
            Neighbors indices (of each query point) among training samples
            self._fit_X.
        Returns
        -------
        local_reachability_density : array, shape (n_samples,)
            The local reachability density of each sample.
        """
        dist_k = self._distances_fit_X_[neighbors_indices, self.n_neighbors_ - 1]
        reach_dist_array = np.maximum(distances_X, dist_k)
        # 1e-10 to avoid `nan' when nb of duplicates > n_neighbors_:
        return 1. / (np.mean(reach_dist_array, axis=1) + 1e-10)