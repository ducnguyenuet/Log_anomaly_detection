from deeplog_trainer.model.model_manager import ModelManager
import pytest


@pytest.fixture(scope='class')
def setup():
    model_manager = ModelManager()
    return model_manager

@pytest.mark.parametrize("model_type, input_size, kwargs, "
                         "expected_output_shape",
                         [(ModelManager.MODEL_TYPE_LOG_KEYS, 10,
                           {'num_tokens': 27, 'lstm_units': 128}, 27),
                          (ModelManager.MODEL_TYPE_LOG_PARAMS, 5,
                           {'num_params': 3}, 3),
                          pytest.param(ModelManager.MODEL_TYPE_LOG_KEYS, 10,
                                       {'num_tokens': '27'}, 27,
                                       marks=pytest.mark.xfail(
                                           raises=ValueError)),
                          pytest.param(ModelManager.MODEL_TYPE_LOG_PARAMS, 10,
                                       {'num_tokens': 3}, 27,
                                       marks=pytest.mark.xfail(
                                           raises=ValueError)),
                          pytest.param('unknown_model', 10,
                                       {'num_tokens': '27'}, 27,
                                       marks=pytest.mark.xfail(
                                           raises=Exception))
                          ]
                         )
def test_model_manager(setup, model_type, input_size, kwargs,
                       expected_output_shape):
    model = setup.build(model_type, input_size, **kwargs)
    if 'lstm_units' in kwargs:
        assert model.layers[1].units == kwargs['lstm_units']
    else:
        assert model.layers[1].units == setup.DEFAULT_LSTM_UNITS
    assert model.output_shape[1] == expected_output_shape
