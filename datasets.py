from abc import ABC
from sklearn import datasets
from keras.datasets import mnist
from keras import backend as K
import keras


def get_dataset(data_json):
    # Uses the dataset name from JSON file to return proper dataset
    dataset = CLASS_NAME.get(data_json["name"])
    if dataset:
        return dataset(data_json)
    else:
        print("Dataset {} not supported".format(data_json['name']))


class Dataset(ABC):
    """Provides the template for all other datasets."""

    def __init__(self, data_json):
        self.name = data_json["name"]

        # Very important this should be hard coded as dictionary for each dataset
        # Link for reference: https://apple.github.io/coremltools/generated/coremltools.converters.keras.convert.html
        self.coreml_specs = {
            'input_names': "",
            'output_names': "",
            # Optional
            # 'class_labels': [""],
        }

        self.batch_size = data_json.get("batch_size")  # Scalar
        self.input_shape = None  # Tuple
        self.output_shape = None # Tuple
        self.loss = data_json["loss"]  # String that describes the Keras loss function e.g. "mse"
        self.metrics = data_json["metrics"]  # List of Keras strings for metrics
        self.train_data = None  # Should be a numpy array
        self.train_labels = None  # Should be a numpy array
        self.test_data = None # Should be a numpy array
        self.test_labels = None # Should be a numpy array
        self.data_split = 0.8 # Train/Test split


class IrisDataset(Dataset):

    def __init__(self, data_json):
        # TODO(ngundotra): rewrite this code, and some test models
        Dataset.__init__(self, data_json)
        # Parse your options from data_json
        # Do any loading that needs to be done here
        iris = datasets.load_iris()
        self.input_shape = [iris.data.shape[1]] # 4
        self.output_shape = [len(iris.target_names)] # 3
        self.epochs = data_json.get("epochs")

        # Split the data into training and testing data
        self.train_data = iris.data[:int(len(iris.data) * self.data_split)]
        self.train_labels = iris.target[:int(len(iris.target) * self.data_split)]
        self.test_data = iris.data[int(len(iris.data) * self.data_split) : len(iris.data)]
        self.test_labels = iris.target[int(len(iris.target) * self.data_split) : len(iris.target)]

        # Set coreml_specs
        self.coreml_specs = {
            "input_names": "iris_features",
            "output_names": iris.target_names.tolist()
        }


class CirclesDataset(Dataset):

    def __init__(self, data_json):
        Dataset.__init__(self, data_json)
        # Parse your options from data_json
        # Do any loading that needs to be done here


class MNISTDataset(Dataset):
    def __init__(self, data_json):
        Dataset.__init__(self, data_json)
        # Parse options
        self.img_rows = data_json.get("img_rows")
        self.img_cols = data_json.get("img_cols")
        self.num_classes = data_json.get("num_classes")
        self.epochs = data_json.get("epochs")
        self.loss = keras.losses.categorical_crossentropy
        self.input_shape = [28, 28, 1]
        self.output_shape = [10]
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
            input_shape = (1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
            input_shape = (self.img_rows, self.img_cols, 1)

        scale = 1/255.0
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train *= scale
        x_test *= scale
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        self.train_data = x_train
        self.train_labels = y_train
        self.test_data = x_test
        self.test_labels = y_test

        # Set CoreML specs
        self.coreml_specs = {
            'input_names': "image",
            'output_names': "digit_probs",
            # Special because pictures & class_labels
            'class_labels': [str(i) for i in range(10)], # ['0', '1', ...]
            'image_input_names': 'image',
            'image_scale': 1/255.0,
        }


# dictionary of dataset classes supported
CLASS_NAME = {
    'Iris': IrisDataset,
    'Circles': CirclesDataset,
    'MNIST': MNISTDataset
}
