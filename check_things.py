from main import make_model, get_json
from modelgraph import ModelGraph


def check_dim(layer, to_check):
    """
    Throws ValueError if shapes mismatch.
    Returns True
    """
    # Convert to list, remove None in the first index
    try:
        shape = layer.output_shape[1:]
    except AttributeError as e:
        shape = layer.shape.as_list()[1:]
    for x, y in zip(shape, to_check):
        if x != y:
            print("expected: {}, received: {}".format(shape, to_check))
            raise ValueError("Shape did not meet expectations")
    return True


def check_model(model, test_dict):
    """
    Only works for sequential models
    """
    check_dim(model.input_layer.layer, test_dict['input_layer'])
    for i in range(len(model.layers)):
        check_dim(model.layers[i].layer, test_dict['layer_{}'.format(i)])
    return True


def run_test(fname, test_dict):
    """
    Takes in a test JSON file, and hand written dictionary that
    checks the output shape of each layer in the model.
    """
    print("Testing the model shape.")
    model = make_model(get_json(fname))
    passed = check_model(model, test_dict)
    print("Did{}pass".format(" " if passed else " NOT "))


if __name__ == "__main__":
    test_dict = {
        'input_layer': (6, 9),
        'layer_0': [54]
    }
    run_test('input_and_flatten.json', test_dict)

