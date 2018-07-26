from SPLayers import InputLayer

def test_input_layer():
    test = {'d0': 6, 'd1': 6, 'd2': 6}
    innie = InputLayer.make_layer(test)
    return innie.d0 == test['d0'] and innie.d1 == test['d1'] and innie.d2 == test['d2']

if __name__ == "__main__":
    print("Input Layer basic test passed: {}".format('T' if test_input_layer() else 'F'))