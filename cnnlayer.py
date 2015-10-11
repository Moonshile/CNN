#coding=utf-8

"""
Input Layer

Author: Kaiqiang Duan (段凯强)
Email:  moonshile@foxmail.com
"""

import numpy

from bplayer import BpLayer
from inputlayer import InputLayer
from utils import conv2d, tanh, dtanh, downsample, upsample

class ConvLayer(BpLayer):

    def __init__(self, rng, connections, conv_size, learning_rate=0.05):
        super(ConvLayer, self).__init__()
        self.connections = connections
        self.theta = rng.uniform(size=(len(connections), conv_size[0], conv_size[1]))
        self.b = rng.uniform(size=len(connections))
        self.learning_rate = learning_rate

    def output(self):
        """
        Generate output of this layer
        """
        self.x_imgs = self.prev_layer.output()
        return numpy.asarray(map(
            lambda i: tanh(self.b[i] + reduce(
                lambda res, j: res + conv2d(self.x_imgs[j], self.theta[i]),
                self.connections[i],
                0
            )),
            xrange(0, len(self.connections))
        ))


    def back_propagate(self):
        """
        Refine parameters of this layer with residuals from next layer
        """
        # compute gradient
        partial_theta = numpy.asarray(map(
            lambda i: numpy.rot90(conv2d(
                reduce(lambda res, x: res + self.x_imgs[x], self.connections[i], 0),
                numpy.rot90(self.delta[i], 2)
            ), 2),
            xrange(0, len(self.connections))
        ))
        parital_b = numpy.asarray(map(lambda x: numpy.sum(x), self.delta))
        # if previous layer is input layer, then do nothing
        if isinstance(self.prev_layer, InputLayer):
            return
        # compute residuals of previous pooling layer
        if not self.prev_layer.connections:
            self.prev_layer.connections = [[] for i in xrange(0, len(self.x_imgs))]
            for i in xrange(0, len(self.connections)):
                for c in self.connections[i]:
                    self.prev_layer.connections[c].append(i)
        conv_full_res = numpy.asarray(map(
            lambda i: conv2d(
                self.delta[i],
                numpy.rot90(self.theta[i], 2),
                border_mode='full'
            ),
            xrange(0, len(self.theta))
        ))
        self.prev_layer.delta = numpy.asarray(map(
            lambda i: dtanh(self.x_imgs[i])*reduce(
                lambda res, x: res + conv_full_res[x],
                self.prev_layer.connections[i],
                0
            ),
            xrange(0, len(self.x_imgs))
        ))
        # update weights and bias
        self.theta -= self.learning_rate*partial_theta
        self.b -= self.learning_rate*parital_b
        # continue back propagating
        self.prev_layer.back_propagate()



class MaxPoolingLayer(BpLayer):

    def __init__(self):
        super(MaxPoolingLayer, self).__init__()

    def output(self):
        """
        Generate output of this layer
        """
        self.x_imgs = self.prev_layer.output()
        return numpy.asarray(map(lambda x: downsample(x), self.x_imgs))


    def back_propagate(self):
        """
        Refine parameters of this layer with residuals from next layer
        """
        # compute residuals of previous convolutional layer
        img_shape = self.x_imgs[0].shape
        self.prev_layer.delta = numpy.asarray(map(lambda d: upsample(d, img_shape), self.delta))
        # continue back propagating
        self.prev_layer.back_propagate()
        


if __name__ == '__main__':

    from PIL import Image

    image_file = open('test.jpg')
    image = Image.open(image_file)
    rng = numpy.random.RandomState()

    il = InputLayer()
    il.set_level(None, None)
    il.set_image(image)
    c_layer0 = ConvLayer(rng, [[0]]*6, (5, 5))
    s_layer0 = MaxPoolingLayer()

    il.set_level(None, c_layer0)
    c_layer0.set_level(il, s_layer0)
    s_layer0.set_level(c_layer0, None)

    s_output0 = s_layer0.output()
    assert s_output0.shape == (6, 25, 63)
    s_layer0.delta = rng.uniform(size=(6, 25, 63))
    s_layer0.back_propagate()

    image_file.close()

