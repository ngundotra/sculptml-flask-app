"""
Parses a JSON spec file for model creation.
Supports training on our custom "Datasets"
Check `main.py` for usage cases.
"""
from SPLayers import DenseLyr, FlattenLyr, ReshapeLyr, InputLyr
from tensorflow import keras
from keras import Sequential, Model

CLASS_NAME = {
    'DenseLyr': DenseLyr,
    # ConvLyr: 'ConvLyr',
    'FlattenLyr': FlattenLyr,
    'ReshapeLyr':ReshapeLyr,
    'InputLyr': InputLyr
    }


class ModelGraph(object):
    """Read file docs"""

    def __init__(self, spec_dict):
        self.spec_dict = spec_dict
        self.name = spec_dict['model_name']
        self.num_layers = spec_dict['num_layers']
        self._create_layers()
        self._compose_model()
        self._compile_model()

    def _create_layers(self):
        """
        Iteratively parses the layers of the JSON
        Asks each layer class to instantiate each layer
        """
        self.layers = []
        prev_out = None
        self.input_layer = InputLyr(self.spec_dict['input_layer'])

        for i in range(self.num_layers):
            curr_layer_spec = self.spec_dict['layer_{}'.format(i)]
            layer_cls = CLASS_NAME[curr_layer_spec['layer']]
            new_layer = layer_cls(curr_layer_spec)
            self.layers.append(new_layer)

    def _compose_model(self):
        """
        Strings the layers together into an actual keras model
        """
        self.model = self.input_layer.layer
        # Stacks the layers together
        for layer in self.layers:
            # Access the Keras layer portion of our custom SPLayers
            self.model = layer.layer(self.model)
            self.model = Model(self.input_layer.layer, self.model)

    def _compile_model(self):
        """
        Compiles the model with a loss metric, though we could probably just strip the graph from
        the tf.Session.... so I don't know if this is necessary, (esp in long run, with customizable
        models)
        """
        self.loss = self.spec_dict['loss']
        self.opt = self.spec_dict['optimizer']
        self.model.compile(self.opt, self.loss)

