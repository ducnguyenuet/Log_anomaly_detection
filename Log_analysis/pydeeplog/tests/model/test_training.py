import pytest
from deeplog_trainer.model.training import ValLossLogger, ModelTrainer
from deeplog_trainer.model.data_preprocess import DataPreprocess
import logging as logger
from tests.model.mocks import MockModel
import json
import numpy as np
from testfixtures import LogCapture


@pytest.fixture(scope='function')
def setup(dataset):
    # List of unique keys in the training file
    vocab = list(set([x for seq in dataset for x in seq]))
    data_preprocess = DataPreprocess(vocab=vocab)
    train_logger = ValLossLogger(logger, loss_index='loss',
                                 metric_index='accuracy')
    model_trainer = ModelTrainer(logger, epochs=50, early_stop=7,
                                 batch_size=512)
    return data_preprocess, train_logger, model_trainer

@pytest.fixture(autouse=True)
def capture():
    with LogCapture() as capture:
        yield capture

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
def test_model_trainer(setup, dataset, capture, window_size=7):
    data_preprocess, train_logger, model_trainer = setup
    dataset = np.array(dataset, dtype=object)
    dataset = data_preprocess.encode_dataset(dataset)
    train_idx, val_idx, test_idx = \
        data_preprocess.split_idx(len(dataset), train_ratio=0.7,
                                  val_ratio=0.85)
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    X_train, y_train = data_preprocess.transform(
        data_preprocess.chunks(train_dataset, window_size=window_size),
        add_padding=window_size
    )
    X_val, y_val = data_preprocess.transform(
        data_preprocess.chunks(val_dataset, window_size=window_size),
        add_padding=window_size
    )
    mocked_model = MockModel()
    train_logger.on_train_begin()
    history = model_trainer.train(mocked_model, [X_train, y_train],
                                  [X_val, y_val])
    train_logger.on_train_end()
    capture.check(
        ('root', 'INFO', 'Start training'),
        ('root', 'INFO', '{}: inf ({}: 0.0000) - Val. {}: inf '
                         '({}: 0.0000)'.format(train_logger.loss_index,
                                               train_logger.metric_index,
                                               train_logger.loss_index,
                                               train_logger.metric_index)),
        ('root', 'INFO', 'Training finished')
    )
    assert isinstance(history, MockModel)
