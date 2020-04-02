# QuickML - Backend

This is the library of tools needed to receive requests for building, training and compiling machine learning models to CoreML models for visualization and education purposes.

# HERES HOW TO RUN THE CODE

```python
python main.py mnist_cnn.json # or 
python main.py iris_spec.json
```



The framework that will ultimately handle all requests can vary, maybe Python Flask, Node.js, even Scala Akka. This should be relatively simple: just building out endpoints
for a simple RESTful API. Internally, calls will probably be routed to an AWS server to build + train + compile the ML models. The last step will be delivering the ML model
to the client. One issue I've read about for transferring large files that need to be written to disk by a client is backpressure. Which is when the client writing to disk is
slower than the rate of data transfer.

To handle this I've read Scala's Akka handles backpressure by doing some "reactive streaming" stuff.

## Framework I propose to handle this:

For networking requests: Node.js (for resume building) or Python Flask (for simplicity)

For ML building + training + compiling: Python (TF) + shell code for compiling the CoreML code

For CoreML transfer: Scala's Akka or something which takes into account a phone's limited ability to read in a CoreML model.
__(Note that this part of the framework depends on the Cocoapods available to handle this)__


## ML Model Building + Training

Python file should take in a JSON file location as an argument. There should be some 2-step process by which the Python app can communicate that 
1) the model was built successfully, and 
2) the model is done training, and 3) the model has been compiled.

For now, we can just flush these updates to the stdout, and try to compress everything into 1 process.

## Model Transferring

We should see if there are any cocoapods that do this...

--- 
# Contributions

Thank you to Newman Hu, Allen Chen, Candace Chiang, and Riley Shanahan for their time, thoughts, and enthusiasm on this project.
