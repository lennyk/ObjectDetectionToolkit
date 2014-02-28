__author__ = 'lenny'

from os import listdir
from os.path import isfile, join
import cv2
import scipy.misc
from find_obj import explore_match
import libraries

class Detection(object):

    def __init__(self):
        # get the available libraries
        self._available_libraries = []
        for library in libraries.Library.__subclasses__():
            self._available_libraries.append(library.__name__)

    def select_file(self, path, name, allOption=False):
        """Prompt the user to select a file from the supplied path."""

        # get files in path
        files = [ f for f in listdir(path) if isfile(join(path,f)) ]

        # copy files and add the "ALL" option if enabled
        options = list(files)
        if allOption:
            options.append('ALL')

        # select an option
        selection = self.select_option(options, name)

        # if it's a file, put the path back on the front
        result = []
        if selection is 'ALL':
            for file in files:
                result.append(join(path, file))
        else:
            result.append(join(path, selection))

        # return the selection
        return result

    def select_option(self, options, name):
        """Prompt the user to select a numbered option from the supplied list."""

        # list the available files
        for (i, option) in enumerate(options):
            print (str(i+1) + ") " + option)

        # prompt user to select a numbered file
        selection = raw_input("select a " + name + ": ")

        # if input is invalid, require user to try again
        while not selection.isdigit() or int(selection) not in range(1, len(options) + 1):
            selection = raw_input("invalid selection. try again: ")
        return options[int(selection) - 1]

    def identify_model(self, library, models, sample):
        """Identifies the model from the list of models in the given sample using the given library."""

        # downsize the sample
        models = [scipy.misc.imresize(model, 0.5) for model in models]
        sample = scipy.misc.imresize(sample, 0.25)

        # find the keypoints and descriptors, result is tuple containing (keypoints, descriptors)
        info_models = [library.find_keypoints_descriptors(model) for model in models]
        info_sample = library.find_keypoints_descriptors(sample)

        # match descriptors
        matches = [library.match_descriptors(info_model[1], info_sample[1]) for info_model in info_models]

        # filter matches and select the best
        filtered_matches_set = [library.filter_matches(match) for match in matches]

        # select the largest bunch of matches and keep track of its index to select the associated model
        best_matches_index, best_matches = max(enumerate(filtered_matches_set), key = lambda tup: len(tup[1]))

        points_model, points_sample, keypoint_pairs =\
            library.prepare_match_points(info_models[best_matches_index][0], info_sample[0], best_matches)

        # old OpenCV sample code to filter matches
        # p1, p2, kp_pairs = filter_matches(kp_sample, kp_model, matches)

        # use OpenCV sample code to display the match
        explore_match('find_obj', models[best_matches_index], sample, keypoint_pairs)

        cv2.waitKey()
        cv2.destroyAllWindows()

    def run(self):
        """Run the main loop."""

        # select a library
        library_name = self.select_option(self._available_libraries, 'library')
        library = getattr(libraries, library_name)()
        models_dir = library.models_dir
        samples_dir = library.samples_dir

        # select and load a single model, or all models with the "ALL" option
        models = self.select_file(models_dir, 'model', True)
        model_images = [cv2.imread(model, 0) for model in models]

        # select and load a sample
        # will receive an array from select_file() but we only ever have one sample so index it
        sample = self.select_file(samples_dir, 'sample')[0]
        sample_image = cv2.imread(sample, 0)

        self.identify_model(library, model_images, sample_image)

if __name__=="__main__":
    Detection().run()