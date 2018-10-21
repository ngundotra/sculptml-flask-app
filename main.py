"""
Receives the filename of a JSON specification for building and training an ML model.
"""
import json
import sys
from modelgraph import ModelGraph
from datasets import get_dataset


def get_json(fname):
    with open(fname, 'rb') as json_f:
        spec_dict = json.load(json_f)
    return spec_dict


def make_model(spec_dict):
    """Returns built tf model + session"""
    return ModelGraph(spec_dict)


def train_model(mg, spec_dict):
    """
    Trains a model on the datasets.
    Saves the model to directory.
    Returns directory location.
    Hands off compilation to shell script.
    """
    # Return Dataset obj
    dataset = get_dataset(spec_dict["dataset"])
    final_acc, test_acc = mg.train_on(dataset)
    mg.save()
    return mg.savedir


def main():
    print("Retrieving json...")
    model_spec = get_json("mnist_cnn.json")
    # get the model part of json
    print("Creating the model...")
    model = make_model(model_spec.get("model"))
    print(model.model.summary())
    # get the dataset part of json
    # dataset = get_dataset(model_spec.get("dataset"))
    saveddir = train_model(model, model_spec)


    print("Train accuracy is:", model.train_acc)
    print("Test accuracy is:", model.test_acc)


if __name__ == '__main__':
    main()
