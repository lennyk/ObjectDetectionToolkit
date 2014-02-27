__author__ = 'lenny'

import abc

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


class Books(Library):

    def __init__(self):
        self._models_dir = 'images/books/models'
        self._samples_dir = 'images/books/samples'