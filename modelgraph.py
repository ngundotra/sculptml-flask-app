"""
Parses a JSON spec file for model creation.
Supports training on our custom "Datasets"
Check `main.py` for usage cases.
"""
# from tensorflow.keras
from keras import Sequential, Model
from datasets import Dataset
import keras
from SPLayers import (
    DenseLyr,
    FlattenLyr,
    ReshapeLyr,
    InputLyr,
    Conv2DLyr,
    MaxPooling2DLyr,
    DropoutLyr)


CLASS_NAME = {
    'DenseLyr': DenseLyr,
    'Conv2DLyr': Conv2DLyr,
    'FlattenLyr': FlattenLyr,
    'ReshapeLyr': ReshapeLyr,
    'InputLyr': InputLyr,
    'MaxPooling2DLyr': MaxPooling2DLyr,
    'DropoutLyr': DropoutLyr,
    'Adadelta': keras.optimizers.Adadelta
}


class ModelGraph(object):
    """Read file docs"""

    def __init__(self, spec_dict):
        self.train_acc = None
        self.test_acc = None
        self.dataset = None
        self.spec_dict = spec_dict
        self.name = spec_dict['model_name']
        self.num_layers = spec_dict['num_layers']
        self._create_layers()
        self._compose_model2()

    def _create_layers(self):
        """
        Iteratively parses the layers of the JSON
        Asks each layer class to instantiate each layer
        """
        self.layers = []
        # prev_out = None
        self.input_layer = InputLyr(self.spec_dict['input_layer'])

        for i in range(self.num_layers):
            curr_layer_spec = self.spec_dict['layer_{}'.format(i)]
            layer_cls = CLASS_NAME[curr_layer_spec['layer']]
            # print(curr_layer_spec)
            # calls SPLayer classes on current layer spec
            new_layer = layer_cls(curr_layer_spec)
            self.layers.append(new_layer)
            # appends instances of each SPlayer class

    def _compose_model(self):
        """
        Strings the layers together into an actual keras model
        """
        # Input class
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
        # start off model, possibly alter the type of model in the future
        self.model = Sequential()
        self.model.add(self.input_layer.input_layer)
        for layer in self.layers:
            self.model.add(layer.layer)

    def _compile_model(self, dataset):
        """
        Compiles the model using the specified dataset with a loss metric and
        verifies that input and output shapes match
        """
        if not isinstance(dataset, Dataset):
            raise ValueError("Dataset should be one of the Dataset classes")
        if self.input_layer.layer_shape != dataset.input_shape and self.model.output_shape != dataset.output_shape:
            raise ValueError("Input or output shapes do not align with input or output shape of dataset", self.input_layer.layer_shape, dataset.input_shape)

        self.loss = dataset.loss
        self.opt = self.spec_dict['optimizer']
        self.metrics = dataset.metrics

        self.model.compile(optimizer=self.opt,
                           loss=self.loss, metrics=self.metrics)

    def train_on(self, dataset):
        """
        Compiles then trains model on the dataset using the options parsed from the JSON
        dataset is a Dataset object
        """
        self.dataset = dataset # Figured would be useful to have this
        self._compile_model(dataset)
        print("batch size: " + str(dataset.batch_size) + ", epochs: "+str(dataset.epochs))
        hist = self.model.fit(dataset.train_data, dataset.train_labels, batch_size=dataset.batch_size, epochs=dataset.epochs)
        self.train_acc = hist.history['acc']
        self.test_acc = self.model.evaluate(dataset.test_data, dataset.test_labels, verbose=0)[1]
        return self.train_acc, self.test_acc


    def save(self):
        # TODO(ramimostafa): Save Keras model to folder with special name and overwrite folder if it exists
        # TODO(ramimostafa): Also setup self.savedir
        return None
