import flask
from flask import Flask, send_file, Response, request
import os, json, sys, requests, base64, pickle
from poing import model_name_from_json
from os.path import exists 
from subprocess import Popen
# PLEASE NOTE THAT ALL THESE COMMANDS ARE MEANT TO BE RUN FROM 
# A VIRTUALENV WITH requirements.txt INSTALLED
# This will insure that `python` maps to the right commands
# Otherwise, all the Popen (subprocess-training & killing) calls will 100% break

# This sets up the flask app to be run like so:
# activate venv before running
# source scultpml-venv/bin/activate
# python server.py
# This will run the server
app = Flask(__name__)

# This is a global variable that keeps track of 
# the process training each model
# proc_map['model_name'] = proc.pid
proc_map = {}

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
    json_fname = 'last_user_request.json'
    with open(json_fname, 'w') as f:
        f.write(json.dumps(json_dict))
    model_name = model_name_from_json(json_fname)
    proc = Popen(["python", 'main.py', json_fname])
    # Save PID with the model's name
    proc_map[model_name] = proc.pid
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
    # path to progress file
    prog_file = model_name + "_progress_log.json"
   
    # If the model exists, send it in json response
    data_dict = {'ready': 0, 'model_path': None}
    if os.path.exists(coreml_path):
        data_dict['progress'] = 1
        data_dict['ready'] = 1 
    elif os.path.exists(prog_file):
        with open(prog_file,"r") as f:
            logs = json.load(f)
        data_dict['progress_info'] = logs
    else:
        print("model not ready")
        data_dict['progress'] = 0
        data_dict['ready'] = 0
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

@app.route('/get-all', methods=['GET']) 
def get_all_models():
    """
    Returns a JSON of # of models
    & each model
        name
        dataset
        size (of coremlmodel)
        train acc
        test acc
    """
    model_names = os.listdir('saved-models/')
    models_info = {'num_models': len(model_names)}
    for i, model_dir in enumerate(model_names):
        cwd = os.path.join('saved-models/', model_dir)
        sizemodel = os.stat(cwd, 'coremlmodel.mlmodel').st_size
        with open(os.path.join(cwd, 'train_info.json'), 'r') as json_f:
            train_info = json.load(json_f)
        info = {
            'name': model_dir,
            'dataset': train_info.get('dataset'),
            'size': sizemodel,
            'train_acc': train_info.get('train_acc'),
            'test_acc': train_info.get('test_acc')
        }
        models_info['model_'+str(i)] = info
    return Response(json.dumps(models_info), status=200, mimetype='application/json')


@app.route('/stop-model', methods=['GET'])
def stop_model():
    """
    Tries to stop model with given name from being trained.
    If the process cannot be found, 404 is thrown
    """
    model_name = request.args['model_name']
    if model_name in proc_map.keys():
        target_pid = proc_map[model_name]
        kill_proc = Popen(["kill", "-2", str(target_pid)])
        # Remove the PID from the map, so we don't accidentally kill other processes
        proc_map.pop(model_name)
        return "Model is being killed", 200
    return "Process ID not found", 404
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)