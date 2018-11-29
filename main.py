"""
Receives the filename of a JSON specification for building and training an ML model.
"""
import json
import sys
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
    dataset = get_dataset(spec_dict["dataset"])
    final_acc, test_acc, did_finish = mg.train_on(dataset)
    savedir = mg.save()
    return final_acc, test_acc, did_finish


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
    from os.path import join
    coreml_model.save(join(mg.savedir, "coremlmodel"))
    return coreml_model


def main(json_query):
    """
    Receives a JSON, builds the corresponding Keras model, trains it, and compiles it
    The last step, `compile_model` saves the compiled model to
    saved-models/
        `model_name`/
            *.coremlmodel
            # The user's JSON req
            user_request.json
            # The info on training (see below)
            train_info.json
    """
    from os.path import join

    print("Retrieving json...")
    print(json_query)
    model_spec = get_json(json_query)
    print(model_spec)
    print("Creating the model...")
    model = make_model(model_spec.get("model"))
    print(model.model.summary())

    train_acc, test_acc, did_finish = train_model(model, model_spec)
    print("Train accuracy is:", train_acc)
    print("Test accuracy is:", test_acc)
    train_info = {
        'train_acc': train_acc, 'test_acc': test_acc, 
        'finished': did_finish, 'dataset': model.dataset.name
    }
    # Write the info to the train_info json
    with open(join(model.savedir, "train_info.json"), "wb") as f:
        json.dump(train_info, f)
    # Write the model spec to 'user_request.json'
    with open(join(model.savedir, 'user_request.json'), 'w') as f:
        json.dump(model_spec, f)

    coreml_model = compile_model(model)


if __name__ == '__main__':
    """Runs main on the flask file passed in as first arg"""
    main(sys.argv[1])
