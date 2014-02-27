__author__ = 'lenny'

import abc
import cv2

class Library(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def models_dir(self):
        return self._models_dir

    @property
    def samples_dir(self):
        return self._samples_dir

    @abc.abstractmethod
    def find_keypoints_descriptors(self, image):
        pass

class Books(Library):

    def __init__(self):
        self._models_dir = 'images/books/models'
        self._samples_dir = 'images/books/samples'
        self._detector = cv2.SIFT()

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        self._matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def find_keypoints_descriptors(self, image):
        return self._detector.detectAndCompute(image, None)

    def match_descriptors(self, des1, des2):
        return self._matcher.knnMatch(des1, des2, k = 2)