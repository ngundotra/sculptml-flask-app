"""
Parses a JSON spec file for model creation.
Supports training on our custom "Datasets"
Check `main.py` for usage cases.
"""
from SPLayers import DenseLyr, FlattenLyr, ReshapeLyr, InputLyr, Conv2DLyr, MaxPooling2DLyr, DropoutLyr
#from tensorflow.keras 
from tensorflow.keras import Sequential, Model

CLASS_NAME = {
    'DenseLyr': DenseLyr,
    'Conv2DLyr': Conv2DLyr,
    'FlattenLyr': FlattenLyr,
    'ReshapeLyr':ReshapeLyr,
    'InputLyr': InputLyr,
    'MaxPooling2DLyr': MaxPooling2DLyr,
    'DropoutLyr': DropoutLyr
    }


class ModelGraph(object):
    """Read file docs"""

    def __init__(self, spec_dict):
        self.spec_dict = spec_dict
        self.name = spec_dict['model_name']
        self.num_layers = spec_dict['num_layers']
        self._create_layers()
        self._compose_model2()
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
            #print(curr_layer_spec)
            #calls SPLayer classes on current layer spec
            new_layer = layer_cls(curr_layer_spec)
            self.layers.append(new_layer)
            #appends instances of each SPlayer class

    def _compose_model(self):
        """
        Strings the layers together into an actual keras model
        """
        #Input class
        self.model = self.input_layer.layer
        # Stacks the layers together
        for layer in self.layers:
            # Access the Keras layer portion of our custom SPLayers
            self.model = layer.layer(self.model)
            print(self.model)
            self.model = Model(self.input_layer.layer, self.model)

    def _compose_model2(self):
        """
        Strings the layers together into an actual keras model
        alternate approach
        """
        #start off model, possibly alter the type of model in the future
        self.model = Sequential()
        self.model.add(self.input_layer.input_layer)

        for layer in self.layers:
            self.model.add(layer.layer)


    def _compile_model(self):
        """
        Compiles the model with a loss metric, though we could probably just strip the graph from
        the tf.Session.... so I don't know if this is necessary, (esp in long run, with customizable
        models)
        """
        self.loss = self.spec_dict['loss']
        self.opt = self.spec_dict['optimizer']
        self.metrics = self.spec_dict['metrics']
        
        self.model.compile(optimizer=self.opt, loss=self.loss, metrics=self.metrics)

