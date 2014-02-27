__author__ = 'lenny'

from os import listdir
from os.path import isfile, join
import cv2
import scipy.misc
# import numpy
from matplotlib import pyplot

from find_obj import filter_matches, explore_match

import numpy as np
import matplotlib.pyplot as plt

import inspect
import libraries

class Detection(object):

    def __init__(self):
        # get the available libraries
        self._available_libraries = []
        for library in libraries.Library.__subclasses__():
            self._available_libraries.append(library.__name__)

    def selectFile(self, path, name):
        """Prompt the user to select a file from path."""

        # get files in path
        files = [ f for f in listdir(path) if isfile(join(path,f)) ]

        # select a file (0 to n-1)
        file = self.selectOption(files, name)

        # return the file
        return join(path, file)

    def selectOption(self, options, name):
        # list the available files
        for (i, option) in enumerate(options):
            print (str(i+1) + ") " + option)
        # require user to select a numbered file
        selection = input("select a " + name + ": ")
        # if input is invalid, require user to try again
        while selection not in range(1, len(options) + 1):
            selection = input("invalid selection. try again: ")
        return options[selection - 1]

    def run(self):
        """Run the main loop."""

        # select a library
        library_name = self.selectOption(self._available_libraries, 'library')
        library = getattr(libraries, library_name)()
        models_dir = library.models_dir
        samples_dir = library.samples_dir

        # select and load a model
        model = self.selectFile(models_dir, 'model')
        model = cv2.imread(model, 0)

        # select and load a sample
        sample = self.selectFile(samples_dir, 'sample')
        sample = cv2.imread(sample, 0)

        # downsize the sample
        sample = scipy.misc.imresize(sample, 0.25)
        model = scipy.misc.imresize(model, 0.5)

        # find the keypoints and descriptors
        kp_sample, des_sample = library.find_keypoints_descriptors(model)
        kp_model, des_model = library.find_keypoints_descriptors(sample)

        # match descriptors
        matches = library.match_descriptors(des_sample, des_model)

        # use copied OpenCV sample code to filter matches and show the match
        p1, p2, kp_pairs = filter_matches(kp_sample, kp_model, matches)
        explore_match('find_obj', model,sample,kp_pairs) # cv2 shows image

        cv2.waitKey()
        cv2.destroyAllWindows()


        # # detect keypoints of model and sample
        # model_pyrdown = cv2.pyrDown(model) # pyrdown
        # sample_pyrdown = cv2.pyrDown(sample)
        # model_gray = cv2.cvtColor(model_pyrdown, cv2.COLOR_RGB2GRAY) # convert to grayscale
        # sample_gray = cv2.cvtColor(sample_pyrdown, cv2.COLOR_RGB2GRAY)
        # surfer = cv2.SURF()
        # model_mask = numpy.uint8(numpy.ones(model_gray.shape))
        # sample_mask = numpy.uint8(numpy.ones(sample_gray.shape))
        # # model_keypoints = surfer.detect(model_gray, model_mask)
        # # sample_keypoints = surfer.detect(sample_gray, sample_mask)
        # model_keypoints, model_descriptors = surfer.detectAndCompute(model_gray,model_mask,
        #                                                                useProvidedKeypoints = True)
        # sample_keypoints, sample_descriptors = surfer.detectAndCompute(sample_gray, sample_mask,
        #                                                                useProvidedKeypoints = True)
        #
        # # FLANN parameters
        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params = dict(checks = 50)
        # matcher = cv2.FlannBasedMatcher(index_params, search_params)
        #
        # matches = matcher.knnMatch(model_descriptors, sample_descriptors, k = 2)
        #
        #
        # p1, p2, kp_pairs = filter_matches(model_keypoints, sample_keypoints, matches)
        # explore_match('find_obj', model_gray, sample_gray, kp_pairs)#cv2 shows image
        #
        # cv2.waitKey()
        # cv2.destroyAllWindows()

if __name__=="__main__":
    Detection().run()