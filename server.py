import flask
from flask import Flask, send_file, Response, request
import os, json, sys, requests, base64 
from os.path import exists
from subprocess import Popen

# This sets up the flask app to be run like so:
# python server.py
# This will run the server
app = Flask(__name__)

@app.route('/')
def home():
    """Our landing page"""
    return "Welcome to SculptML!"

@app.route('/make-model', methods=['POST'])    
def start_btc_process():
    """
    This starts the training process.
    """
    json_dict = request.get_json()
    print("Received JSON", file=sys.stderr)
    json_fname = 'user_request.json'
    with open(json_fname, 'w') as f:
        f.write(json.dumps(json_dict))
    proc = Popen(["python", 'main.py', json_fname])
    return Response("Request received", 200)

@app.route("/check-model", methods=['GET'])
def check_model():
    """
    Checks for coreml model progress
    Accepts GET arg: {'model_name'}
    If the model is not ready then it will return
    {'ready': 0, 'progress': [0,1)}
    If the model is ready, then the JSON looks like
    {'ready': 1, 'progres': 1}
    """
    # Load the model path
    model_name = request.args['model_name']
    coreml_location = 'saved-models/'+model_name
    coreml_path = coreml_location + '/model.h5'

    # If the model exists, send it in json response
    data_dict = {'ready': 0, 'model_path': None}
    if os.path.exists(coreml_path):
        data_dict['progress'] = 1
        data_dict['ready'] = 1 
    return Response(json.dumps(data_dict), status=200, mimetype='application/json')

@app.route('/get-model', methods=["GET"])
def get_model():
    """
    Either sends back binary coremlmodel, or 404 not found
    """
    model_name = request.args['model_name']
    coreml_location = 'saved-models/'+model_name
    coreml_path = coreml_location + '/coremlmodel.mlmodel'

    binary_model = None
    if os.path.exists(coreml_path):
        with open(coreml_path, 'rb') as f:
            binary_model = f.read()
        return Response(binary_model, status=200, content_type='application/octet')
    else:
        return "Model not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)