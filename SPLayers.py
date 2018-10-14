from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import ( 
    Input, 
    InputLayer,
    Dense, 
    Reshape, 
    Conv2D, 
    Flatten, 
    MaxPooling2D, 
    Dropout
)


class SPModelLayer(ABC):
    """
    Maybe extend(?) Idk
    """


def parse_tuple(str_tup):
    """
    Parses a string like '(1, 2, 3)' into actual tuple containing (1, 2, 3)
    ...does not work on negative numbers...
    """
    vec = []
    curr_num = 0
    for c in str_tup:
        if '0' <= c <= '9':
            curr_num *= 10
            curr_num += int(c)
        if c == ',' or c == ')':
            vec.append(curr_num)
            curr_num = 0
    return tuple(vec)


def shrink_tuple(vec):
    """Gets rid of leftmost elements <= 0"""
    shrunk = []
    for elm in vec:
        if elm > 0:
            shrunk.append(elm)
    return shrunk


class InputLyr(SPModelLayer):
    """
    Dummy class. Used to verify that datasets & transfer models agree on input shapes. Probably a good place to
    add tests to ensure that data is being transferred properly between modular pieces...
    """

    def __init__(self, model_spec):
        layer_shape = model_spec['dim']
        if len(layer_shape) < 3:
            raise ValueError("Dim passed had < 3 fields. We expect 3")
        layer_shape = parse_tuple(layer_shape)
        layer_shape = shrink_tuple(layer_shape)
        self.model_spec = model_spec
        self.layer_shape = layer_shape
        self.layer = Input(layer_shape)
        #extra input layer for compose_model2
        self.input_layer = InputLayer(layer_shape)


class DenseLyr(SPModelLayer):
    def __init__(self, model_spec):
        self.model_spec = model_spec
        self.units = model_spec['units']
        self.layer = Dense(self.units, activation=model_spec['activation'])


class ReshapeLyr(SPModelLayer):
    def __init__(self, model_spec):
        layer_shape = model_spec['dim']
        if len(layer_shape):
            raise ValueError("Expect dim to be = 3")
        self.layer_shape = shrink_tuple(parse_tuple(layer_shape))
        self.model_spec = model_spec
        self.layer = Reshape(self.layer_shape)


class FlattenLyr(SPModelLayer):
    def __init__(self, model_spec):
        self.model_spec = model_spec
        self.layer = Flatten()


#TODO: implement convolutional layer, dropout, and batch normalization
class Conv2DLyr(SPModelLayer):
    def __init__(self, model_spec):
        self.model_spec = model_spec
        self.filters = model_spec['filters']
        self.kernel_size = parse_tuple(model_spec['kernel_size'])
        self.activation = model_spec['activation']
        self.layer = Conv2D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation)

        
class MaxPooling2DLyr(SPModelLayer):
    def __init__(self, model_spec):
        self.model_spec = model_spec
        self.pool_size = parse_tuple(model_spec['pool_size'])
        self.layer = MaxPooling2D(pool_size=self.pool_size)


class DropoutLyr(SPModelLayer):
    def __init__(self, model_spec):
        self.model_spec = model_spec
        self.rate = model_spec['rate']
        self.layer = Dropout(rate=self.rate)
