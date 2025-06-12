import pytest
from tests.model.mocks import MockModel
from deeplog_trainer.model.model_evaluator import ModelEvaluator
from deeplog_trainer.model.data_preprocess import DataPreprocess
import json
import numpy as np

@pytest.fixture(scope='function')
def setup(dataset):
    # List of unique keys in the training file
    vocab = list(set([x for seq in dataset for x in seq]))
    data_preprocess = DataPreprocess(vocab=vocab)
    return data_preprocess

def get_data(min_length=4):
    filepath = 'data/data.json'
    dataset = []
    with open(filepath, 'r') as fp:
        data = json.load(fp)
        for d in data['data']:
            seq = d['template_seq']
            if len(seq) < min_length:
                # Skip short sequences
                continue
            dataset.append(seq)
        yield dataset

@pytest.mark.parametrize("dataset", get_data())
def test_predict(setup, dataset, window_size=7):
    data_preprocess = setup
    dataset = np.array(dataset, dtype=object)
    dataset = data_preprocess.encode_dataset(dataset)
    train_idx, val_idx, test_idx = \
        data_preprocess.split_idx(len(dataset), train_ratio=0.7,
                                  val_ratio=0.8)
    val_dataset = dataset[val_idx]
    X_val, y_val = data_preprocess.transform(
        data_preprocess.chunks(val_dataset, window_size=window_size),
        add_padding=window_size
    )
    predict_output = np.zeros((2, np.shape(y_val)[1]))
    # Create two random probability vectors of length the number of unique
    # tokens
    for i, j in enumerate([0.15, 0.05, 0.1, 0.7]):
        predict_output[0][4*i] = j
        predict_output[1][3*i+2] = j
    # Mock the model and set the previous vectors as output of the function
    # model.predict
    mocked_model = MockModel(predict_output=predict_output)
    model_evaluator = ModelEvaluator(mocked_model, top_k=4)
    # Take a sample of the first two lines of the validation data to test the
    # methods of the class ModelEvaluator.
    # The indexes with the highest probability value must be the same of vectors
    # randomly generated before.
    assert (model_evaluator.predict(X_val[0:2, :, :]) ==
            np.array([[4, 8, 0, 12], [5, 8, 2, 11]])).all()
    anomalies_idx = model_evaluator.get_anomalies_idx(X_val[0:2, :, :],
                                                      y_val[0:2, :])
    # The second sequence is well predicted since it predicts the key 11, which
    # has the highest probability to come out (0.7)
    # The first one instead it doesn't, hence the method get_anomalies_idx must
    # output just the first index 0.
    assert anomalies_idx == [0]
    scores = model_evaluator.compute_scores(X_val[0:2, :, :], y_val[0:2, :])
    assert scores['n_items'] == 2
    assert scores['n_correct'] == 1
    # Since one sequence out of two is correctly predicted, the accuracy value
    # must be worth 0.5
    assert scores['accuracy'] == 0.5
