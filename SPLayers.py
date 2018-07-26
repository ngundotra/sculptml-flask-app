from abc import ABC, abstractmethod, abstractclassmethod
import torch

class SPModelLayer(ABC):
    """
    Sets up basic functions all layers should have.
    Each layer should really be its own PyTorch Model.
    It's nice because we can set up our own defaults, customize layer implementations, and then implement new layers
    when necessary. Also cool because we can use this modular approach to do transfer learning, I think...
    """
    @abstractclassmethod
    def make_layer(self, json_dict):
        pass

    @abstractmethod
    def get_output_shape(self):
        pass

    @abstractmethod
    def get_input_shape(self):
        pass

class InputShape:
    """
    For interpolating between layers :/
    """

    def __init__(self, dict):
        self.d0 = dict['d0']
        self.d1 = dict['d1']
        self.d2 = dict['d2']

    def __iter__(self):
        return iter([self.d0, self.d1, self.d2])

    def __repr__(self):
        return 'shape: ({}, {}, {})'.format(self.d0, self.d1, self.d2)

    def __eq__(self, other):
        if isinstance(other, InputShape):
            return self.d0 == other.d0 and self.d1 == other.d1 and self.d2 == other.d2
        return False

    def __copy__(self):
        return InputShape({'d0': self.d0, 'd1': self.d1, 'd2': self.d2})


class InputLayer(SPModelLayer):
    """
    Dummy class. Used to verify that datasets & transfer models agree on input shapes. Probably a good place to
    add tests to ensure that data is being transferred properly between modular pieces...
    """

    def __init__(self, in_dict):
        self.input_shape = InputShape(in_dict)

    @classmethod
    def make_layer(cls, json_dict):
        return InputLayer(json_dict)

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.get_input_shape()


class DenseLayer(SPModelLayer, torch.nn.Module):

    def __init__(self, params):
        torch.nn.Module.__init__(self)
        self.in_shape = ['']
        self.units = params['units']
        self.layer = torch.nn.Linear( )

    @classmethod
    def make_layer(self, json_dict):
        return DenseLayer(json_dict)

    def get_input_shape(self):
        return
