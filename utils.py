#coding=utf-8

"""
Some useful functions and structures
Author: Kaiqiang Duan (段凯强)
Email:  moonshile@foxmail.com
"""

import numpy
import itertools

def conv2d(image, filter, border_mode='valid'):
    """
    Perform a convolution for 2-dimension discrete data

    :param border_mode: {'valid', 'full'}, ‘valid’ only apply filter to
                        complete patches of the image. Generates output
                        of shape: image_shape - filter_shape + 1.
                        ‘full’ zero-pads image to multiple of filter shape to
                        generate output of shape: image_shape + filter_shape - 1.
    """
    h, w = image.shape
    a, b = filter.shape
    assert border_mode in {'valid', 'full'}
    cond = lambda m, n, i, j: m - i < 0 or n - j < 0 or m - i >= h or n - j >= w
    conv_at = lambda m, n: sum(sum(
        filter[i, j]*(0 if cond(m, n, i, j) else image[m - i, n - j]) for j in xrange(0, b)
    ) for i in xrange(0, a))
    return numpy.asarray([[
            conv_at(row, col) for col in xrange(
                b - 1 if border_mode == 'valid' else 0,
                w if border_mode == 'valid' else w + b - 1
            )
        ] for row in xrange(
            a - 1 if border_mode == 'valid' else 0,
            h if border_mode == 'valid' else h + a - 1
        )
    ])

def downsample(image, size=(2,2), ignore_border=False):
    """
    Perform a max pooling for 2-dimension data
    """
    row, col = image.shape
    a, b = size
    if ignore_border:
        row_steps = [(i, i + a) for i in xrange(0, row - row%a, a)]
        col_steps = [(i, i + b) for i in xrange(0, col - col%b, b)]
    else:
        row_steps = [(i, i + a) for i in xrange(0, row, a)]
        col_steps = [(i, i + b) for i in xrange(0, col, b)]
    sub_regions = itertools.product(row_steps, col_steps)
    return numpy.reshape(
        map(
            lambda ((ra, rb), (ca, cb)): numpy.max(image[ra:rb, ca:cb]),
            sub_regions
        ),
        (len(row_steps), len(col_steps))
    )

def cut_image(image, shape):
    a, b = shape
    return image[:a, :b]

def extend_image(image, shape):
    row, col = numpy.shape(image)
    a, b = shape
    res = [list(line) + [list(line)[-1]]*(a - row) for line in list(image)]
    res += [res[-1]]*(b - col)
    return numpy.asarray(res)

def upsample(image, shape, size=(2,2), border_mode='cut'):
    """
    Perform the reverse of a max pooling for 2-dimension data

    :param border_mode: {'cut', 'extend'}, 'cut' to cut the result to fit shape
                        'extend' to extend the result to fit shape
    """
    row, col = image.shape
    a, b = size
    assert border_mode in {'cut', 'extend'}
    res = numpy.asarray([
        [image[i/a, j/b] for j in xrange(0, col*b)]
        for i in xrange(0, row*a)]
    )
    return cut_image(res, shape) if border_mode == 'cut' else extend_image(res, shape)

def sigmoid(x):
    return 1./(1. + numpy.exp(-x))

def dsigmoid(y):
    return y*(1. - y)

def tanh(x):
    return numpy.tanh(x)

def dtanh(y):
    return 1. - y**2







if __name__ == '__main__':
    input = numpy.asarray([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    filter = numpy.asarray([
        [1, 1],
        [1, 1]
    ])
    assert (conv2d(input, filter) == numpy.asarray([[12, 16], [24, 28]])).all()
    assert (conv2d(input, filter, border_mode='full') == numpy.asarray(
        [[1, 3, 5, 3], [5, 12, 16, 9], [11, 24, 28, 15], [7, 15, 17, 9]]
    )).all()
    assert sigmoid(10) - 0.999954602131 < 1e-6
    not_ignored = downsample(input)
    assert (not_ignored == numpy.asarray([[5, 6], [8, 9]])).all()
    assert (upsample(not_ignored, (3, 3)) == numpy.asarray(
        [[5, 5, 6], [5, 5, 6], [8, 8, 9]]
    )).all()
    ignored = downsample(input, ignore_border=True)
    assert (ignored == numpy.asarray([[5]])).all()
    assert (upsample(ignored, (3, 3), border_mode='extend') == numpy.asarray([
            [5, 5, 5],
            [5, 5, 5],
            [5, 5, 5]
    ])).all()
    print 'All test cases for utils passed'

