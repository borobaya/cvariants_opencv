
import brisk_constants
from centroids import centroids
from kmeans import KMeans


def _features(cv_image, image_info=None):
    import cv2  # local import to prevent global libdc1394 errors

    """ Takes a cvMat grayscale image and return the corresponding
        brisk descriptors. Do not change the brisk parameters """
    brisk = cv2.BRISK(brisk_constants.BRISK_THRESHOLD,
                      brisk_constants.BRISK_OCTAVES,
                      brisk_constants.BRISK_SCALE)

    try:
        keypoints, descriptors = brisk.detectAndCompute(cv_image, None)
    except:
        return []

    """ Wrap this in a try/except to catch None and empty numpy arrays. It's very hard
        to check for both conditions when the array comes from C, this is the safest way """
    try:
        descriptors_n = len(descriptors)
    except:
        return []

    return descriptors


def _kmeans(descriptors):
    """ Map a set of brisk descriptors (each descriptor is 64 floats) into a
        pre-training cluster space. Each descriptor maps to an integer (i.e.
        the cluster id) so we transform the (n,64) descriptor matrix into
        an n-vector. A k-histogram of the n-vector is returned.  """

    kmeans_model = KMeans(centroids=centroids)

    cluster_ids = [kmeans_model.classify(d) for d in descriptors]

    """ Take the multiplicity of each cluster id in the mapped descriptor space.
        The n-vector of cluster ids becomes a histogram of cardinality 128 """
    histogram = [0] * brisk_constants.N_CLUSTERS

    for cluster_id in set(cluster_ids):
        histogram[cluster_id] = cluster_ids.count(cluster_id)

    return histogram
