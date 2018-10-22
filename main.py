"""
Receives the filename of a JSON specification for building and training an ML model.
"""
import json
import sys
import pdb
from modelgraph import ModelGraph
from datasets import get_dataset
import coremltools


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
    # TODO: will be returning savedir once that is implemented
    return final_acc, test_acc


def compile_model(mg):
    """
    Converts our modelgraph object to coreml model
    :param mg: ModelGraph
    :param spec_dict: Dictionary representing the outermost JSON
    :return: location of saved coremlmodel
    """
    keras_model = mg.model
    dataset = mg.dataset
    desc = dataset.coreml_specs
    coreml_model = coremltools.converters.keras.convert(keras_model, **desc)
    # Assumes mg.savedir is set
    coreml_model.save(mg.savedir)
    return coreml_model


def main():
    """
    Receives a JSON, builds the corresponding Keras model, trains it, and compiles it
    """
    print("Retrieving json...")
    model_spec = get_json(sys.argv[1])
    print("Creating the model...")
    model = make_model(model_spec.get("model"))
    print(model.model.summary())

    # get the dataset part of json
    # dataset = get_dataset(model_spec.get("dataset"))
    train_acc, test_acc = train_model(model, model_spec)
    print("Train accuracy is:", train_acc)
    print("Test accuracy is:", test_acc)

    pdb.set_trace()
    coreml_model = compile_model(model)


if __name__ == '__main__':
    main()
