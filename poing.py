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
    """Asks for `model_name` at the /get-model endpoint"""
    resp = requests.get(server, params={'model_name': model_name})
    return resp

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
