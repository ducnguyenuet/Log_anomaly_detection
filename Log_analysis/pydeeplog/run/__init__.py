from argparse import ArgumentParser
import logging
import sys
import os
import json
import numpy as np
from zipfile import ZipFile

from deeplog_trainer.model.data_preprocess import DataPreprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

ZIPFILE_APP_MODEL = 'deeplog_app_model.zip'
ZIPFILE_CORE_MODEL = 'deeplog_core_model.zip'

def create_datasets(logger, input_path, min_length, train_ratio, val_ratio):
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dataset = []
    # Load the data parsed from Drain:
    with open(os.path.join(root_path, input_path, 'data.json'), 'r') as fp:
        data = json.load(fp)
        for d in data['data']:
            seq = d['template_seq']
            if len(seq) < min_length:
                # Skip short sequences
                continue
            dataset.append(seq)
    dataset = np.array(dataset, dtype=object)
    # List of unique keys in the training file
    vocab = list(set([x for seq in dataset for x in seq]))
    data_preprocess = DataPreprocess(vocab=vocab)
    dataset = data_preprocess.encode_dataset(dataset)
    train_idx, val_idx, test_idx = \
        data_preprocess.split_idx(len(dataset), train_ratio=train_ratio,
                                  val_ratio=val_ratio)
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    logger.info(
        'Datasets sizes: {}, {}, {}'.format(len(train_idx), len(val_idx),
                                            len(test_idx)))
    return train_dataset, val_dataset, test_dataset, data_preprocess

def zip_directory(temp_path, zip_filepath, skip_files=None):
    if skip_files is None:
        skip_files = []
    with ZipFile(zip_filepath, 'w') as zip_obj:
        # Iterate over all the files in the directory
        for folder_name, sub_folders, filenames in os.walk(temp_path):
            for filename in filenames:
                if filename in skip_files:
                    continue
                filepath = os.path.join(folder_name, filename)
                # Add file to zipfile
                zip_obj.write(filepath, filename)

def add_workflows_runner_args(parser: ArgumentParser):
    """
    Arguments for runner of workflows.
    """
    parser.add_argument("--input_path", type=str,
                        help="Put the filepath from root folder where Drain"
                             "results are saved",
                        default='artifacts/drain_result')
    parser.add_argument("--output_path", type=str,
                        help="Put the path of the output directory",
                        default='artifacts/workflows')
    parser.add_argument("--window_size", type=int,
                        help="Put the window_size parameter", default=10)
    parser.add_argument("--train_ratio", type=float,
                        help="Put the percentage of dataset size to define the"
                             "train set", default=0.7)
    parser.add_argument("--val_ratio", type=float,
                        help="Put the percentage of dataset size to define the"
                             "validation set", default=0.85)
    parser.add_argument("--threshold", type=float,
                        help="Put the similarity threshold", default=0.8)
    parser.add_argument("--back-steps", type=int,
                        help="Put the number of steps backwards to research"
                             "similar workflows", default=1)

def add_drain_runner_args(parser: ArgumentParser):
    """
    Arguments for Drain runner
    """
    parser.add_argument("--input_file", type=str,
                        help="Put the input filepath from root folder")
    parser.add_argument("--output_path", type=str,
                        help="Put the name of the directory where the results "
                             "will be saved",
                        default='artifacts/drain_result')
    parser.add_argument("--config_file", type=str,
                        help="Put the filepath of the config file")
    parser.add_argument("--window_size", type=int,
                        help="Put the window_size parameter", default=10)

def add_log_key_model_runner_args(parser: ArgumentParser):
    """
    Arguments for runner of log keys (i.e. Drain templates) model.
    """
    parser.add_argument("--input_path", type=str,
                        help="Put the filepath from root folder where Drain"
                             "results are saved",
                        default='artifacts/drain_result')
    parser.add_argument("--output_path", type=str,
                        help="Put the name of the directory where the results "
                             "will be saved",
                        default='artifacts/log_key_model_result')
    parser.add_argument("--config_file", type=str,
                        help="Put the filepath of the config file")

    parser.add_argument("--window_size", type=int,
                        help="Put the window_size parameter", default=10)
    parser.add_argument("--lstm_units", type=int,
                        help="Put the number of units in each LSTM layer",
                        default=64)
    parser.add_argument("--max_epochs", type=int,
                        help="Put the maximum number of epochs if the process "
                             "is not stopped before by the early_stop",
                        default=50)
    parser.add_argument("--train_ratio", type=float,
                        help="Put the percentage of dataset size to define the"
                             "train set", default=0.7)
    parser.add_argument("--val_ratio", type=float,
                        help="Put the percentage of dataset size to define the"
                             " validation set", default=0.85)
    parser.add_argument("--early_stop", type=int,
                        help="Put the number of epochs with no improvement "
                             "after which training will be stopped", default=7)
    parser.add_argument("--batch_size", type=int,
                        help="Put the number of samples that will be propagated"
                             " through the network", default=512)
    parser.add_argument("--out_tensorboard_path", type=str,
                        help="Put the name of the folder where to save the "
                             "tensorboard results if desired", default=None)
    parser.add_argument("--top_k", type=int,
                        help="Put the number of top candidates to estimate the "
                             "number of anomalies", default=9)

def add_parameters_model_runner_args(parser: ArgumentParser):
    """
    Arguments for runner of anomaly detection model in parameter values.
    """
    parser.add_argument("--input_path", type=str,
                        help="Put the filepath from root folder where Drain"
                             "results are saved")
    parser.add_argument("--output_path", type=str,
                        help="Put the path of the output directory",
                        default='artifacts/log_par_model_result')
    parser.add_argument("--window_size", type=int,
                        help="Put the window_size parameter", default=5)
    parser.add_argument("--lstm_units", type=int,
                        help="Put the number of units in each LSTM layer",
                        default=64)
    parser.add_argument("--max_epochs", type=int,
                        help="Put the maximum number of epochs if the process "
                             "is not stopped before by the early_stop",
                        default=100)
    parser.add_argument("--train_ratio", type=float,
                        help="Put the percentage of dataset size to define the"
                             "train set", default=0.5)
    parser.add_argument("--val_ratio", type=float,
                        help="Put the percentage of dataset size to define the"
                             " validation set", default=0.75)
    parser.add_argument("--early_stop", type=int,
                        help="Put the number of epochs with no improvement "
                             "after which training will be stopped", default=7)
    parser.add_argument("--batch_size", type=int,
                        help="Put the number of samples that will be propagated"
                             " through the network", default=16)
    parser.add_argument("--out_tensorboard_path", type=str,
                        help="Put the name of the folder where to save the "
                             "tensorboard results if desired", default=None)
    parser.add_argument("--alpha", type=float, help="confidence level of the "
                                                    "confidence interval",
                        default=0.95)
