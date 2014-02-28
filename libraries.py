__author__ = 'lenny'

import abc
import cv2
import numpy

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

    def find_keypoints_descriptors(self, image):
        pass

    def prepare_match_points(self, keypoints_model, keypoints_sample, matches):
        matched_keypoints_model, matched_keypoints_sample = [], []
        for match in matches:
            matched_keypoints_model.append(keypoints_model[match[0].trainIdx])
            matched_keypoints_sample.append(keypoints_sample[match[0].queryIdx])
        points_model = numpy.float32([keypoint.pt for keypoint in matched_keypoints_model])
        points_sample = numpy.float32([keypoint.pt for keypoint in matched_keypoints_sample])
        keypoint_pairs = zip(matched_keypoints_model, matched_keypoints_sample)
        return points_model, points_sample, keypoint_pairs

class Books(Library):

    def __init__(self):
        self._models_dir = 'images/books/models'
        self._samples_dir = 'images/books/samples'
        self._detector = cv2.SIFT()
        self._MIN_MATCH_COUNT = 10

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        self._matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def find_keypoints_descriptors(self, image):
        return self._detector.detectAndCompute(image, None)

    def match_descriptors(self, des1, des2):
        return self._matcher.knnMatch(des2, des1, k=2)

    def filter_matches(self, matches, ratio=0.7):
        good_matches = []
        for match in matches:
            if len(match) == 2 and match[0].distance < ratio*match[1].distance:
                good_matches.append(match)
        return good_matches

class Autos(Books, Library):

    def __init__(self):
        super(Autos, self).__init__()
        self._models_dir = 'images/autos/models'
        self._samples_dir = 'images/autos/samples'