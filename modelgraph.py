"""
Parses a JSON spec file for model creation.
Supports training on our custom "Datasets"
Check `main.py` for usage cases.
"""
from SPLayers import DenseLyr, ConvLyr, FlattenLyr, ReshapeLyr, InputLyr
CLASSES = {
    DenseLyr: 'DenseLyr',
    ConvLyr: 'ConvLyr',
    FlattenLyr: 'FlattenLyr',
    ReshapeLyr: 'ReshapeLyr',
    InputLyr: 'input_lyr'
    }

class ModelGraph(object):
    """Read file docs"""

    def __init__(self, spec_dict):
        self.spec_dict = spec_dict
        self.name = spec_dict['model_name']
        self.num_layers = spec_dict['num_layers']
        self._create_layers()

    def _create_layers(self):
        """
        Iteratively parses the layers of the JSON
        Asks each layer class to instantiate each layer
        """
        self.layers = []
        prev_out = None
        input_layer = InputLyr.make(self.spec_dict[CLASSES[InputLyr]])

        for i in range(self.num_layers):
            curr_layer_spec = self.spec_dict['layer_{}'.format(i)]
            layer_cls = CLASSES[curr_layer_spec['layer']]
            new_layer = layer_cls.make(curr_layer_spec, prev=prev_out)
            prev_out = new_layer.out_tensor
            self.layers.append(new_layer)


