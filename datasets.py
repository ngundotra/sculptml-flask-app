from abc import ABC
from sklearn import datasets

def get_dataset(data_json):
    # Uses the dataset name from JSON file to return proper dataset
    if data_json["name"] == "iris":
        return IrisDataset(data_json)
    if data_json["name"] == "circles":
        return CirclesDataset(data_json)


class Dataset(ABC):

    def __init__(self, data_json):
        self.name = data_json["name"]
        self.batch_size = None  # Scalar
        self.input_shape = None  # Tuple
        self.loss = None  # String that describes the Keras loss function e.g. "mse"
        self.metrics = ["accuracy"]  # List of Keras strings for metrics
        self.train_data = None  # Should be a numpy array
        self.train_labels = None  # Should be a numpy array
        self.test_data = None # Should be a numpy array
        self.test_labels = None # Should be a numpy array

class IrisDataset(Dataset):

    def __init__(self, data_json):
        Dataset.__init__(data_json)
        # Parse your options from data_json
        # Do any loading that needs to be done here
        iris = datasets.load_iris()
        self.input_shape = iris.data.shape

        # Split the data into training and testing data
        # Suggestion: Add a parameter for train/test split?
        self.train_data = iris.data[:len(iris.data) // 2]
        self.train_labels = iris.target[:len(iris.target) // 2]
        self.test_data = iris.data[len(iris.data) // 2 : len(iris.data)]
        self.test_labels = iris.target[len(iris.target) // 2 : len(iris.target)]

class CirclesDataset(Dataset):

    def __init__(self, data_json):
        Dataset.__init__(data_json)
        # Parse your options from data_json
        # Do any loading that needs to be done here


