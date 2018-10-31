"""Utilities for spamming the test server"""
import requests
import json

def post_up(server, fname):
    """Sends shit to the /make-model endpoint"""
    if not isinstance(server, str) or not isinstance(fname, str):
        raise ValueError("fname and server should be strings")
    url = server + '/make-model'
    with open(fname, 'rb') as json_f:
        resp = requests.post(url, json=json.load(json_f))
    return resp
    
def get_model(server, model_name):
    """Asks for `model_name` at the /get-model endpoint
    
    e.g. get_model('http://127.0.0.1:5000', model_name_from_json('iris_spec.json'))

    returns 404 not found if model not ready, or byte_data of the model (probably need to write to file first before using)
    """
    url = server + '/get-model'
    resp = requests.get(url, params={'model_name': model_name})
    return resp

def check_model(server, model_name):
    """Asks for progress update at the /check-model endpoint.
    You should always check for model before GET-ing a model to avoid 404 errors

    e.g. check_model('http://127.0.0.1:5000', model_name_from_json('iris_spec.json'))
    returns {'ready': 0 or 1, 'progress': [0, 1]}
    """
    url = server + '/check-model'
    resp = requests.get(url, params={'model_name': model_name})
    return resp

def get_model_local(fname):
    """Gets model using model_name from a specific json file"""
    return get_model('http://127.0.0.1:5000', model_name_from_json('iris_spec.json'))

def model_name_from_json(fname):
    """Just a nice util. Fname is name of JSON file"""
    with open(fname, 'rb') as f:
        spec = json.load(f)
    return spec['model']['model_name']

if __name__ == '__main__':
    url = 'http://127.0.0.1:5000'
    fname = 'iris_spec.json'
    resp = post_up(url, fname)
    print(resp)
