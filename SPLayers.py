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
        # specifies default argument if argument is not in json spec
        get_arg = lambda arg, value: model_spec[a] if model_spec.get(a) else v
        
        self.filters = model_spec['filters']
        self.kernel_size = parse_tuple(model_spec['kernel_size'])
        self.strides = parse_tuple(get_arg('strides', '(1,1)'))
        self.padding = get_arg('padding', 'valid')
        self.data_format = get_arg('data_format', None)
        self.dilation_rate = parse_tuple(get_args('dilation_rate', '(1,1)'))
        self.activation = get_arg('activation', None)
        self.use_bias = get_arg('use_bias', True)
        self.kernel_initializer = get_arg('kernal_initializer', 'glorot_uniform')
        self.bias_initializer = get_arg('bias_initializer', 'zeros')
        self.kernel_regularizer = get_arg('kernel_regularizer', None)
        self.bias_regularizer = get_arg('bias_regularizer', None)
        self.activity_regularizer = get_arg('activity_regularizer', None)
        self.kernel_constraint = get_arg('kernel_contraint', None)
        self.bias_constraint = get_arg('bias_contraint', None)
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
        # specifies default argument if argument is not in json spec
        get_arg = lambda arg, value: model_spec[a] if model_spec.get(a) else v

        self.pool_size = parse_tuple(get_arg('pool_size', '(2,2'))
        self.strides = get_arg('strides', None)
        self.padding = get_arg('padding', 'valid')
        self.data_format = get_arg('data_format', None)
        self.layer = MaxPooling2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format
        )


class DropoutLyr(SPModelLayer):
    def __init__(self, model_spec):
        self.model_spec = model_spec
        # specifies default argument if argument is not in json spec
        get_arg = lambda arg, value: model_spec[a] if model_spec.get(a) else v
        
        self.rate = model_spec['rate']
        self.noise_shape = get_arg('noise_shape', None)
        self.seed = get_arg('seed', None)
        self.layer = Dropout(
            rate=self.rate,
            noise_shape=self.noise_shape,
            seed=self.seed
        )


class BatchNormalizationLyr(SPModelLayer):
    def __init__(self, model_spec):
        self.model_spec = model_spec
        # specifies default argument if argument is not in json spec
        get_arg = lambda arg, value: model_spec[a] if model_spec.get(a) else v

        self.axis = get_arg('axis', -1)
        self.momentum = get_arg('momentum', 0.99)
        self.epsilon = get_arg('epsilon', 1e-3)
        self.center = get_arg('center', True)
        self.scale = get_arg('scale', True)
        self.beta_initializer = get_arg('beta_initializer', 'zeros')
        self.gamma_initializer = get_arg('gamma_initializer', 'zeros')
        self.moving_mean_initializer = get_arg('moving_mean_initializer', 'zeros')
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