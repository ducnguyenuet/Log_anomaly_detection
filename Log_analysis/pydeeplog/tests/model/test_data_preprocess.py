from deeplog_trainer.model.data_preprocess import DataPreprocess
import pytest
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
        return dataset

def get_seqs(min_length=4):
    filepath = 'data/data.json'
    with open(filepath, 'r') as fp:
        data = json.load(fp)
        for d in data['data']:
            seq = d['template_seq']
            if len(seq) < min_length:
                # Skip short sequences
                continue
            yield seq

@pytest.mark.parametrize("dataset, train_ratio, val_ratio",
                         [(get_data(), 0.7, 0.85)])
def test_split_idx(dataset, setup, train_ratio, val_ratio):
    train_idx, val_idx, test_idx = \
        setup.split_idx(len(dataset), train_ratio, val_ratio)
    assert (train_idx == range(int(len(dataset) * train_ratio))).all()
    assert (val_idx == range(int(len(dataset) * train_ratio),
                             int(len(dataset) * val_ratio))).all()
    assert (test_idx == range(int(len(dataset) * val_ratio),
                              len(dataset))).all()

@pytest.mark.parametrize("seq", get_seqs())
def test_chunks_seq(mocker, seq, window_size=7):
    mocked_vocab = mocker.patch.object(DataPreprocess, '__init__',
                                       return_value=None)
    data_preprocess = DataPreprocess(mocked_vocab)
    list_of_chunks = data_preprocess.chunks_seq(seq, window_size=window_size)
    if len(seq) > window_size:
        assert np.shape(list_of_chunks) == (len(seq) - window_size + 1,
                                        window_size)
    else:
        assert np.shape(list_of_chunks) == (1, len(seq))

@pytest.mark.parametrize("dataset", [(get_data())])
def test_transform(dataset, setup, window_size=7):
    dataset = np.array(dataset, dtype=object)
    dataset = setup.encode_dataset(dataset)
    assert setup.get_dictionaries()[1]['[PAD]'] == 0
    chunks = setup.chunks(dataset, window_size=window_size)
    X, y = setup.transform(chunks, add_padding=window_size)
    assert X.shape == (len(chunks), window_size,
                       setup.get_num_tokens())
    assert y.shape == (len(chunks), setup.get_num_tokens())
