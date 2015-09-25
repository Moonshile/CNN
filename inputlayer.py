#coding=utf-8

"""
Input Layer

Author: Kaiqiang Duan (段凯强)
Email:  moonshile@foxmail.com
"""

import numpy

from bplayer import BpLayer

class InputLayer(BpLayer):

    def __init__(self):
        super(BpLayer, self).__init__()

    def set_image(self, image):
        self.x_imgs = (numpy.asarray(image.convert('L'), dtype='float64') - 128)*.8/128.

    def output(self):
        """
        Generate output of this layer
        """
        return numpy.asarray([self.x_imgs])

    def back_propagate(self):
        pass

if __name__ == '__main__':
    from PIL import Image
    image_file = open('test.jpg')
    image = Image.open(image_file)
    il = InputLayer()
    il.set_level(None, None)
    il.set_image(image)
    out = il.output()
    assert out.shape == (1, 53, 130)
    image_file.close()
    
