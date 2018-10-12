"""
Receives the filename of a JSON specification for building and training an ML model.
"""
import json
import sys
from modelgraph import ModelGraph
# from datasets import Dataset, get_dataset


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
    dataset = spec_dict['dataset']
    # Return Dataset obj or something
    dataset = get_dataset(dataset)
    final_acc, test_acc = mg.train_on(dataset)
    mg.save()
    return mg.savedir


if __name__ == '__main__':
    print("Retrieving json...")
    #json_spec = "model_spec_alt.json"
    json_spec = "mnist_cnn.json"
#    model_spec = get_json(sys.argv[1])
    model_spec = get_json(json_spec)
    #print(model_spec)
    print("Creating the model...")
    model = make_model(model_spec)
    print(model.model.summary())
    print(model.model)

