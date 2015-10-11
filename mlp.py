#coding=utf-8

"""
Input Layer

Author: Kaiqiang Duan (段凯强)
Email:  moonshile@foxmail.com
"""

from bplayer import BpLayer
from utils import tanh, dtanh

class HiddenLayer(BpLayer):
    def __init__(self, rng, n_neuron, learning_rate=0.05):
        super(HiddenLayer, self).__init__()
        self.rng = rng
        self.n_neuron = n_neuron
        self.learning_rate = learning_rate

    def output(self):
        """
        Generate output of this layer
        """
        self.x = self.prev_layer.output()
        if not self.theta:
            self.theta = self.rng.uniform(size=(self.n_neuron, len(self.x)))
            self.b = self.rng.uniform(size=(self.n_neuron,))
        return tanh(numpy.dot(self.theta, self.x) + self.b)

    def back_propagate(self):
        """
        Refine parameters of this layer with residuals from next layer
        """
        # compute gradient
        partial_theta = numpy.dot(self.delta, self.x.transpose())
        partial_b = self.delta
        # compute residulas of previous layer, which is a rasterization layer
        # partial of rasterization is unit vector
        self.prev_layer.delta = numpy.dot(self.theta.transpose(), self.delta)
        # update weights and bias
        self.theta -= self.learning_rate*partial_theta
        self.b -= self.learning_rate*partial_b
        # continue back propagating
        self.prev_layer.back_propagate()






