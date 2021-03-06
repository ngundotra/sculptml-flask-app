"""
Parses a JSON spec file for model creation.
Supports training on our custom "Datasets"
Check `main.py` for usage cases.
"""
# from tensorflow.keras
import keras
from keras import Sequential, Model
from datasets import Dataset
import os
from os.path import join, exists
# from model_progress import Model_Progress
import pickle
import json
from SPLayers import (
    DenseLyr,
    FlattenLyr,
    ReshapeLyr,
    InputLyr,
    Conv2DLyr,
    MaxPooling2DLyr,
    DropoutLyr
)


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

    def create_progress_callback(self, total_epochs):
        """
        Create the progress loggin callbacks to dump the model progress into a json for checkmodel get requests
        """
        file_name = self.name + "_progress_log.json"
        def write_json(d):
            data = json.dumps(d)
            with open(file_name,"w") as f:
                f.write(data)
        # write_json({"epoch": 0, "total_epochs": total_epochs})
        return keras.callbacks.LambdaCallback(
            on_epoch_end = lambda epoch, logs: write_json({**logs, "epoch" : epoch, "total_epochs" : total_epochs}),
            on_train_end = lambda logs : os.remove(file_name)
        )

    def train_on(self, dataset):
        """
        Compiles then trains model on the dataset using the options parsed from the JSON
        dataset is a Dataset object
        """

        self.dataset = dataset # Figured would be useful to have this
        self._compile_model(dataset)

        print("batch size: " + str(dataset.batch_size) + ", epochs: "+str(dataset.epochs))
        progress_loggin_callback = self.create_progress_callback(dataset.epochs)
        try: 
            hist = self.model.fit(dataset.train_data, dataset.train_labels, batch_size=dataset.batch_size, epochs=dataset.epochs, 
                callbacks= [progress_loggin_callback])
            self.train_acc = hist.history['acc']
            self.test_acc = self.model.evaluate(dataset.test_data, dataset.test_labels, verbose=0)[1]
            self.finished = 1
        except KeyboardInterrupt:
            # KeyboardInterrupt == Ctrl-C == SIGINT == kill -2
            print("Training has been killed.")
            self.train_acc = 0
            self.test_acc = 0
            self.finished = 0
        return self.train_acc, self.test_acc, self.finished

    def save(self):
        """
        Saves Keras model in the saved-models folder using the keras save function
        Name of the model is the name of the directory
        e.g. model name is "vrab" ==>
            saved-models/
                vrab/
                    model.h5
                    coremlmodel.coremlmodel
        """
        base = 'saved-models'
        if not exists(base):
            os.mkdir(base)
        self.savedir = join(base, self.name)
        if not exists(self.savedir):
            os.mkdir(self.savedir)

        path = join(self.savedir, 'model.h5')
        self.model.save(path)
        return self.savedir
