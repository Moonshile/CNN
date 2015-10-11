#coding=utf-8

"""
Input Layer

Author: Kaiqiang Duan (段凯强)
Email:  moonshile@foxmail.com
"""

from bplayer import BpLayer

class Rasterize(BpLayer):

    def __init__(self, arg):
        super(Rasterize, self).__init__()

    def output(self):
        """
        Generate output of this layer
        """
        self.x_imgs = self.prev_layer.output()
        return self.x_imgs.reshape((reduce(lambda res, x: res*x, self.x_imgs.shape, 1),))

    def back_propagate(self):
        """
        Refine parameters of this layer with residuals from next layer
        """
        self.prev_layer.delta = self.delta.reshape(self.x_imgs.shape)
        self.prev_layer.back_propagate()
        


