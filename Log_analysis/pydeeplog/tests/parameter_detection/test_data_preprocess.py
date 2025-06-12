import pytest
import json
import numpy as np
from deeplog_trainer.parameter_detection.data_preprocess import DataPreprocess


def get_dataset():
    filepath = 'data/dataset.json'
    with open(filepath, 'r') as fp:
        dataset = json.load(fp)['data']
    dataset = np.array(dataset)
    return dataset

def get_expected_arrays():
    expected_X = \
        [[[-0.22187100307898197, -16.537374581073315, -3.216113167869395],
         [0.5607337309470196, 27.797775416477776, 20.375053322044742],
         [0.23121924991718473, 0, 0]],
         [[0.5607337309470196, 27.797775416477776, 20.375053322044742],
         [0.23121924991718473, 17.924300025123188, 8.16357694840985],
         [-1.7108714709057902, 0, 0]],
         [[0.23121924991718473, 17.924300025123188, 8.16357694840985],
         [-1.7108714709057902, -113.78345464504565, -64.9111407649389],
         [-1.0256659856744073, 0, 0]]]
    expected_y = \
        [[0.23121924991718473, 17.924300025123188, 8.16357694840985],
         [-1.7108714709057902, -113.78345464504565, -64.9111407649389],
         [-1.0256659856744073, -61.623178288848585, -29.541163835772842]]
    return expected_X, expected_y

@pytest.mark.parametrize("dataset, train_ratio, val_ratio",
                         [(get_dataset(), 0.7, 0.85)])
def test_split_idx(dataset, train_ratio, val_ratio):
    data_preprocess = DataPreprocess()
    train_idx, val_idx, test_idx = \
        data_preprocess.split_idx(len(dataset), train_ratio, val_ratio)
    assert (train_idx == range(int(len(dataset) * train_ratio))).all()
    assert (val_idx == range(int(len(dataset) * train_ratio),
                             int(len(dataset) * val_ratio))).all()
    assert (test_idx == range(int(len(dataset) * val_ratio),
                              len(dataset))).all()

@pytest.mark.parametrize("sample_dataset, window_size, expected_X, expected_y",
                         [(get_dataset()[0:5], 3, get_expected_arrays()[0],
                           get_expected_arrays()[1])])
def test_data_preprocess(sample_dataset, window_size, expected_X, expected_y):
    data_preprocess = DataPreprocess()
    X, y = data_preprocess.generate(sample_dataset, window_size)
    assert X.shape == (len(sample_dataset)-window_size+1, window_size,
                       sample_dataset.shape[1])
    assert y.shape == (len(sample_dataset)-window_size+1,
                       sample_dataset.shape[1])
    assert (X == expected_X).all()
    assert (y == expected_y).all()
