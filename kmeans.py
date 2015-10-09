import random

import numpy as np
from numpy import linalg

import brisk_constants


class KMeans(object):

    """
    Basic streaming kmeans implementation with kmean++ initialisation.

    Once the model is trained (i.e. centroids are passed into the constructor the model can no
    longer be updated

    """

    def __init__(self, k=brisk_constants.N_CLUSTERS, centroids=None):
        """ k: number of clusters
            centroids: pre-training centroids. Centroids must of dimension k """
        if centroids is not None:
            self.centroids = centroids
            self.counts = None
            self.centroids_update = None
        else:
            self.centroids = np.array([])
            self.counts = [0] * k
            self.centroids_update = [0] * k

        self.init_features = []
        self.k = k

    @property
    def _initialised(self):
        """True if centroids have been initialised."""
        return self.centroids.size > 0

    @property
    def _ready_for_init(self):
        """True if we have enought examples to initialise."""
        magic_number = 10  # from trident-ml implementation
        return len(self.init_features) > self.k * magic_number

    def _init_if_possible(self, features):
        if self._ready_for_init:
            self._init_centroids()
        else:
            self.init_features.append(features)

    def _nearest_centroid(self, features):
        """Return nearest centroid to `features`"""
        return np.min(linalg.norm(features - self.centroids, axis=1))

    def _init_feature_distances(self):
        """
        Find cumulative distance between each of the init_features and their closest centroid.

        See kmeans++ paper for details

        """
        distances = np.array([
            np.power(self._nearest_centroid(fi), 2) for fi in self.init_features
        ])
        return distances.cumsum()

    def _init_centroids(self):
        """
        Initialise centroids with cached features.

        See: k-means++ - Arthur, D. and Vassilvitskii, S. (2007).

        """
        random.shuffle(self.init_features)

        self.centroids = np.array([self.init_features.pop()])
        self.counts[0] = 1

        for ki in range(1, self.k):
            dx = self._init_feature_distances()
            r = random.random() * dx[-1]

            # get index of first element in dx for which dx[i] >= r
            cid = np.where(dx >= r)[0][0]

            self.centroids = np.vstack([self.centroids, self.init_features[cid]])
            self.counts[ki] += 1

            del self.init_features[cid]

        """ After cluster centroids have been choosen, update the model
        with the remaining init_features """
        while self.init_features:
            self.update(self.init_features.pop())

    def update(self, features):
        """
        Update kmeans model if it has been initialised.

        If not store features until we have enought to initialise the model

        """
        if self.counts is None or self.centroids_update is None:
            raise RuntimeError("Cannot update a trained model")

        if not self._initialised:
            self._init_if_possible(features)
            return

        cid = self.classify(features)

        self.counts[cid] += 1
        self.centroids_update[cid] = (features - self.centroids[cid]) * (1.0 / self.counts[cid])
        self.centroids[cid] = self.centroids[cid] + self.centroids_update[cid]

        return cid

    def update_delta(self):
        """
        Return the delta of the last centroid update for each cluster.

        The sum and std
        of the deltas can be used as a *very* rough measure of convergence

        """
        if self.counts is None or self.centroids_update is None:
            raise RuntimeError("Cannot get deltas a trained model")

        if self._initialised:
            return np.sum(np.power(np.array(self.centroids_update), 2), axis=1)
        else:
            return np.array([])

    def classify(self, features):
        """find closest centroid to `features` and return that centroid's index (cluster ID)"""

        if not self._initialised:
            raise RuntimeError("Model is not ready for classification. We have no centroids")

        distances = linalg.norm(features - self.centroids, axis=1)
        cluster_id = np.argmin(distances)
        return cluster_id
