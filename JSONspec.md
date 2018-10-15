JSON Spec

Pretty simple: Model information, like loss, optimizer, metrics,
etc, goes into the top layer of the JSON. Layer information goes
into layer dictionaries (JSON child objs). 

**Note**: `num_layers` must be set to the number of non-input layers.

**Note**: `dim` is passed as a string, which gets parsed to proper dim size

**Note**: The layers are 0-indexed after the input layer. All models are assumed to be
sequential.
The layers must have the correct `layer` name to be parsed correctly. We only support
Dense, Reshape, Flatten, and Conv2D layers.

_Layers_:

- input_layer
    - `dim` is a tuple (6,6) or (6,) or (6,6,6), all of which
    get parsed correctly
- DenseLyr
    - `units` is the output dimension of this layer
    - `activation` supported
- FlattenLyr
    - No args
- ReshapeLyr
    - `dim` reshape outshape
- Conv2DLyr
    - NYI (not yet implemented)
    - See implementation in `SPLayers.py`
    
Example:

```
{
  "__info__": "This is a JSON file used for testing the model creation process",
  "model": {
      "model_name": "Test-v0", # Required
      "num_layers": 1, # Required
      "optimizer": "rmsprop",
      "input_layer": {
        "dim": "(6, 6, 6)"
      },
      "layer_0": {
        "layer": "DenseLyr",
        "units": 128,
        "activation": "relu"
      }
      # other layers here...
  }
  "dataset": {
       "name": "Circles", # Required
       "data_split": 0.8, # Default, not required
       "inner_radius": 10,
       "num_samples": 1000
  }
}
```

