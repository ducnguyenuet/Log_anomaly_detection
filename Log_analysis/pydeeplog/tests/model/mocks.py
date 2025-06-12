import numpy as np


class MockModel:
    def __init__(self, predict_output=None):
        self.predict_output = predict_output

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data, epochs,
            batch_size, callbacks, shuffle, verbose):
        return self

    def predict(self, X_data):
        return self.predict_output
