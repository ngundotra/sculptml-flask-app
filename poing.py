"""Utilities for spamming the test server"""
import requests
import json

def post_up(server, fname):
    if not isinstance(server, str) or not isinstance(fname, str):
        raise ValueError("fname and server should be strings")
    resp = requests.post(server, fname)
    return resp
    
def get_model(server, model_name):
    resp = requests.get(server, params={'model_name': model_name})
    return resp

def model_name_from_json(fname):
    with open(fname, 'rb') as f:
        spec = json.load(f)
    return spec['model']['model_name']

if __name__ == '__main__':
    url = 'http://127.0.0.1:5000'
    fname = 'iris_spec.json'
    resp = post_up(url, fname)
    print(resp)
