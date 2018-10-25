import flask
from flask import Flask, send_file, Response, request
import os, json, sys, requests
from os.path import exists
from subprocess import Popen

# This sets up the flask app to be run like so:
# $ export FLASK_APP=server.py
# $ flask run
# If you're on windows lol here's a link: http://flask.pocoo.org/docs/1.0/quickstart/
app = Flask(__name__)
"""If dummy==True, then flask server *should* return coreml model,

resp = poing.post_up('http://127.0.0.1:5000', 'mnist_cnn.json')
resp
^ will get dummy model"""
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

    @app.route("/get-model", methods=['POST', 'GET'])
    def get_model():
        """
        Checks for coreml model existence, and sends back in JSON if it exists
        Accepts GET arg: {'model_name'}
        If the model is not ready then it will return
        {'ready': False, 'model': None}
        If the model is ready, then the JSON looks like
        {'ready': True, 'model': binary(coremlmodel)}
        """
        # Load the model path
        from main import get_json
        if request.method == 'GET':
            model_name = request.args['model_name']
        if request.headers['Content-Type'] == 'application/json':
            spec = get_json(request.json)
            model_name = spec['model']['model_name']
        coreml_path = 'saved-models/'+model_name+'/model.h5'

        # If the model exists, send it in json response
        data_dict = {'ready': False, 'model': None}
        if os.path.exists(coreml_path):
            with open(coreml_path, 'rb') as f:
                data_dict['model'] = f.read()
            data_dict['ready'] = True
        resp = Response(data_dict, 200)
        return resp

