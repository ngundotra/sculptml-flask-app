from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras
from keras.layers import InputLayer, Dense, Reshape, Conv2D, Flatten


class SPModelLayer(ABC):
    """
    Maybe extend(?) Idk
    """
    @classmethod
    def make(cls, json_dict):
        return cls.__init__(json_dict)


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


class InputLayer(SPModelLayer):
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
        self.layer = InputLayer(input_shape=(layer_shape))


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
