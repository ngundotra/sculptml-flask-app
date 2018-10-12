from main import make_model, get_json
from modelgraph import ModelGraph

import numpy as np

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.backend import clear_session

from sklearn import datasets

def create_model():
    model = make_model(get_json("iris_spec.json"))
    return model.model

def main():
  clear_session()
  iris = datasets.load_iris()

  X = iris.data
  Y = iris.target

  estimator = KerasClassifier(build_fn=create_model)
  estimator.fit(X,Y)
  
  # Right now just predict on the trained data (accuracy metrics should match)
  predictions = estimator.predict(X)

  accuracy = 0
  for i in range(0, len(Y)):
    if (predictions[i] == Y[i]):
      accuracy += 1

  accuracy = accuracy / len(Y)

  print(accuracy)

if __name__ == "__main__": 
    main()
