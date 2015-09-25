#coding=utf-8

"""
Artificial Neurons layer with back propagation

Author: Kaiqiang Duan (段凯强)
Email:  moonshile@foxmail.com
"""

import abc

class BpLayer(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(BpLayer, self).__init__()

    def set_level(self, prev_layer, next_layer):
        self.prev_layer = prev_layer
        self.next_layer = next_layer

    @abc.abstractmethod
    def output(self):
        """
        Generate output of this layer
        """
        pass

    @abc.abstractmethod
    def back_propagate(self):
        """
        Refine parameters of this layer with residuals from next layer
        """
        pass
