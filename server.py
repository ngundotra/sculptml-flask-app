import flask
from flask import Flask, send_file, Response, request
import os, json, sys, requests, base64 
from os.path import exists
from subprocess import Popen

# This sets up the flask app to be run like so:
# $ export FLASK_APP=server.py
# $ flask run
# If you're on windows lol here's a link: http://flask.pocoo.org/docs/1.0/quickstart/
app = Flask(__name__)
"""If dummy==True, then flask server *should* return coreml model,

local_server = 'http://127.0.0.1:5000'
resp = poing.post_up(, 'mnist_cnn.json')
resp = poing.get_model(')
^ will get dummy mo"""
dummy = False # sets up some fake shit, lmao
local = False
if local:
    pypath = '/Users/ngundotra/anaconda3/envs/keras/bin/python'
else:
    # pypath = os.getcwd()
    # print(os.listdir(pypath))
    # pypath = os.path.join(pypath, 'sculptml-venv/bin/python')
    pypath = 'python'


@app.route('/')
def home():
    """Our landing page"""
    return "Welcome to SculptML!"

if dummy:
    # Send MobileNet model, and do not actually run our code
    DUMMY_MODEL = "MobileNet.mlmodel"

    def download_dummy_model():
        """
        Downloads the 16.4 MB MobileNet file we're gonna
        dummy send to users until we set up our pipeline fully.
        """
        url = "https://s3-us-west-2.amazonaws.com/coreml-models/MobileNet.mlmodel"
        if exists(DUMMY_MODEL):
            return None
        response = requests.get(url)
        with open(DUMMY_MODEL, 'wb') as model_file:
            model_file.write(response.content)
        return response


    @app.route('/make-model', methods=['POST', 'GET'])
    def send_model():
        """
        This is a dummy function that returns a pre compiled
        CoreML model to anyone who POSTs to us.
        When we flesh out our CoreML portion of the pipeline,
        then we can replace this with their compiled, custom, CoreML model
        """
        download_dummy_model()
        # MIME Type - types of files recognized by https for compression
        #           * app/octet is basically unknown binary data
        # as_attachment - when file is obtained, it will be downloaded
        #                 instead of being shown inline (if that was an option)
        return send_file(DUMMY_MODEL, mimetype='application/octet-stream',
                as_attachment=True)
else:
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
        proc = Popen([pypath, 'main.py', json_fname])
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