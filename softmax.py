#coding=utf-8

"""
Input Layer

Author: Kaiqiang Duan (段凯强)
Email:  moonshile@foxmail.com
"""

from bplayer import BpLayer
from utils import tanh, dtanh

class Softmax(BpLayer):
    def __init__(self, rng, n_class, learning_rate=0.05):
        super(Softmax, self).__init__()
        self.rng = rng
        self.n_class = n_class
        self.learning_rate = learning_rate

    def output(self):
        """
        Generate output of this layer
        """
        self.x = self.prev_layer.output()
        if not self.theta:
            self.theta = self.rng.uniform(size=(self.n_class, len(self.x)))
            self.b = self.rng.uniform(size=(self.n_class,))
        parts = numpy.exp(numpy.dot(self.theta, self.x) + self.b)
        return parts/sum(parts)

    def back_propagate(self, t, y):
        """
        Refine parameters of this layer with residuals from next layer

        :param t: the class vector of current input
        """
        # delta of this layer
        self.delta = (y - t)*(y - y*y)
        # compute gradient
        partial_theta = numpy.dot(self.delta, self.x.transpose())
        partial_b = self.delta
        # compute residulas of previous layer, i.e., the mlp layer
        self.prev_layer.delta = numpy.dot(self.theta.transpose(), self.delta)*dtanh(self.x)
        # update
        self.theta -= self.learning_rate*partial_theta
        self.b -= self.learning_rate*partial_b
        # continue back propagating
        self.prev_layer.back_propagate()




