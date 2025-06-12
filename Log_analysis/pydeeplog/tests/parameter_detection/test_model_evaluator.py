import pytest
import numpy as np
import logging as logger
import json
from sklearn.preprocessing import MinMaxScaler
from deeplog_trainer.parameter_detection.model_evaluator import ModelEvaluator


def get_data():
    filepath = 'data/dataset.json'
    with open(filepath, 'r') as fp:
        dataset = json.load(fp)['data']
    dataset = np.array(dataset)
    y_true = dataset[0:5]
    y_pred = dataset[5:10]
    return y_true, y_pred

@pytest.mark.parametrize("y_true, y_pred", [(get_data()[0], get_data()[1])])
def test_plot_time_series(mocker, y_true, y_pred):
    model_evaluator = ModelEvaluator(logger)
    sc_X = MinMaxScaler()
    label = 'Test plot'
    timestamps = np.arange(len(y_pred))
    # Mock inverse_transform method of the class MinMaxScaler in sklearn
    mock_scaler = mocker.patch(
        'sklearn.preprocessing.MinMaxScaler.inverse_transform')
    # Random transformation
    mock_scaler.side_effect = [y_true*10, y_pred*10]
    # Mock pyplot.show of the library matplotlib and assert that the method was
    # called once
    mock_plot = mocker.patch('matplotlib.pyplot.show')
    model_evaluator.plot_time_series(sc_X, timestamps, y_true, y_pred, label)
    mock_plot.assert_called()

@pytest.mark.parametrize("y_true, y_pred, mocked_interval",
                         [(get_data()[0], get_data()[1], [-0.5, 0.5])])
def test_build_confidence_intervals(mocker, y_true, y_pred, mocked_interval):
    # Mock norm.interval method of the library scipy.stats and return a random
    # list with two numbers between -1 and 1
    model_evaluator = ModelEvaluator(logger)
    mock_stats_normal = mocker.patch('scipy.stats.norm.interval')
    mock_stats_normal.return_value = mocked_interval
    interval, mse_val = model_evaluator.build_confidence_intervals(y_true,
                                                                   y_pred)
    assert interval == mocked_interval
    assert len(mse_val) == len(y_true)

@pytest.mark.parametrize("interval, mse_val", [([-0.5, 0.5], [0.1, 0.2, 0.3])])
def test_plot_confidence_intervals(mocker, interval, mse_val):
    model_evaluator = ModelEvaluator(logger)
    mock_plot = mocker.patch('matplotlib.pyplot.show')
    model_evaluator.plot_confidence_intervals(interval, mse_val)
    mock_plot.assert_called()

@pytest.mark.parametrize("y_true, y_pred, interval, mocked_mse_test, "
                         "expected_anomalies_idx",
                         [(get_data()[0], get_data()[1], [-0.1, 0.1],
                          [0.1, -0.15, 0.05, -0.05, 0.2], [1, 4])])
def test_get_anomalies_idx(mocker, y_true, y_pred, interval, mocked_mse_test,
                           expected_anomalies_idx):
    model_evaluator = ModelEvaluator(logger)
    mocker.patch(
        'deeplog_trainer.parameter_detection.model_evaluator.ModelEvaluator'
        '.get_mses', return_value=mocked_mse_test)
    anomalies_idx, mse_test = model_evaluator.get_anomalies_idx(y_true, y_pred,
                                                                interval)
    assert mse_test == mocked_mse_test
    assert anomalies_idx == expected_anomalies_idx

@pytest.mark.parametrize("interval, mse_test", [([-0.5, 0.5], [0.1, 0.2, 0.3])])
def test_plot_test_errors(mocker, interval, mse_test):
    model_evaluator = ModelEvaluator(logger)
    mock_plot = mocker.patch('matplotlib.pyplot.show')
    model_evaluator.plot_test_errors(interval, mse_test)
    mock_plot.assert_called()
