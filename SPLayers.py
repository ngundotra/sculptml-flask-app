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
    Dropout,
    BatchNormalization
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
        # extra input layer for compose_model2
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


class Conv2DLyr(SPModelLayer):
    def __init__(self, model_spec):
        self.model_spec = model_spec
        self.filters = model_spec['filters']
        self.kernel_size = parse_tuple(model_spec['kernel_size'])
        self.strides = model_spec['strides']
        self.padding = model_spec['padding']
        self.data_format = model_spec['data_format']
        self.dilation_rate = parse_tuple(model_spec['dilation_rate'])
        self.activation = model_spec['activation']
        self.use_bias = model_spec['use_bias']
        self.kernel_initializer = model_spec['kernel_initializer']
        self.bias_initializer = model_spec['bias_initializer']
        self.kernel_regularizer = model_spec['kernel_regularizer']
        self.bias_regularizer = model_spec['bias_regularizer']
        self.activity_regularizer = model_spec['activity_regularizer']
        self.kernel_constraint = model_spec['kernel_constraint']
        self.bias_constraint = model_spec['bias_constraint']
        self.layer = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint
        )


class MaxPooling2DLyr(SPModelLayer):
    def __init__(self, model_spec):
        self.model_spec = model_spec
        self.pool_size = parse_tuple(model_spec['pool_size'])
        self.strides = model_spec['strides']
        self.padding = model_spec['padding']
        self.data_format = model_spec['data_format']
        self.layer = MaxPooling2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format
        )


class DropoutLyr(SPModelLayer):
    def __init__(self, model_spec):
        self.model_spec = model_spec
        self.rate = model_spec['rate']
        self.layer = Dropout(rate=self.rate)


class BatchNormalizationLyr(SPModelLayer):
    def __init__(self, model_spec):
        self.model_spec = model_spec
        # specifies default argument if argument is not in json spec
        get_arg = lambda arg, value: model_spec[a] if model_spec.get(a) else v

        self.axis = model_spec['axis'] if model_spec.get('axis') else -1
        self.momentum = model_spec['momentum'] if model_spec.get('momentum') else 0.99
        self.epsilon = model_spec['epsilon'] if model_spec.get('epsilon') else 1e-3
        self.center = model_spec['center'] if model_spec.get('center') else True
        self.scale = model_spec['scale'] if model_spec.get('scale') else True
        self.beta_initializer = model_spec['beta_initializer'] if model_spec.get('beta_initializer') else = 'zeros'
        self.gamma_initializer = model_spec['gamma_initializer'] if model_spec.get('gamma_initializer') else = 'ones'
        self.moving_mean_initializer = model_spec['moving_mean_initializer'] if model_spec.get('moving_mean')
        self.moving_variance_initializer = get_arg('moving_variance_initializer', 'zeros')
        self.beta_regularizer = get_arg('beta_regularizer', None)
        self.gamma_regularizer = get_arg('gamma_regularizer', None)
        self.beta_contraint = get_arg('beta_contraint', None)
        self.gamma_contraint = get_arg('gamma_contraint', None)
        self.layer = BatchNormalization(
            axis=self.axis,
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            beta_initializer=self.beta_initializer,
            gamma_initializer=self.gamma_initializer,
            moving_mean_initializer=self.moving_mean_initializer,
            moving_variance_initializer=self.moving_variance_initializer,
            beta_regularizer=self.beta_regularizer,
            gamma_regularizer=self.gamma_regularizer,
            beta_contraint=self.beta_contraint,
            gamma_constraint=self.gamma_contraint
        )