import numpy as np


class ModelEvaluator:
    """
    Description: During detection stage, given an historical sequence of
    integers in input, the LSTM model outputs, thanks to the activation
    function "softmax", a probability distribution over the n log key
    classes. Lastly, fixed the integer parameter top_k, DeepLog detects the
    testing key as an anomaly if it's not among the "top_k" keys with the
    greatest probability value of being the next key, and as normal
    otherwise.
    """
    def __init__(self, model, top_k):
        """
        Attributes:
        :param model: model to be evaluated
        :param top_k: top k candidates parameter
        """
        self.model = model
        self.top_k = top_k

    def get_anomalies_idx(self, X_data, y_data):
        """
        Returns the indexes of testing keys labeled as anomalous during
        detection stage
        """
        n_items = len(X_data)
        y_pred = self.predict(X_data)
        anomalies_idx = [i for i in range(n_items) if
                         np.argmax(y_data[i]) not in y_pred[i]]
        return anomalies_idx

    def predict(self, X_data):
        """Returns  the top_k indexes with the highest probability value"""
        # For each index between 1 and the number of
        # tokens, return the probability value of being the next key of the
        # previous sequence
        y_pred = self.model.predict(X_data)
        y_pred = y_pred.argsort()[:, -self.top_k:]
        return y_pred

    def compute_scores(self, X_data, y_data):
        """
        Returns a dictionary with the number of items parsed, the number of keys
        labeled as normal and the accuracy value.
        """
        n_items = len(X_data)
        n_correct = n_items - len(self.get_anomalies_idx(X_data, y_data))
        accuracy = n_items and n_correct / n_items
        return {
            'n_items': n_items,
            'n_correct': n_correct,
            'accuracy': accuracy,
        }
