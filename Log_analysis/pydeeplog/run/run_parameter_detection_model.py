import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging
import argparse
import json

from deeplog_trainer.parameter_detection.data_preprocess import DataPreprocess
from deeplog_trainer.model.training import ModelTrainer
from deeplog_trainer.model.model_manager import ModelManager
from deeplog_trainer.parameter_detection.model_evaluator import ModelEvaluator

from . import *


def run_parameter_detection_model(logger, input_file, output_path, window_size,
                                  LSTM_units, max_epochs, train_ratio,
                                  val_ratio, early_stop, batch_size,
                                  out_tensorboard_path, alpha):
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # Upload the data to be evaluated
    with open(os.path.join(root_path, input_file), 'r') as fp:
        dataset = json.load(fp)['data']
    dataset = np.array(dataset)
    num_params = dataset.shape[1]
    data_preprocess = DataPreprocess()
    train_idx, val_idx, test_idx = \
        data_preprocess.split_idx(len(dataset), train_ratio=train_ratio,
                                  val_ratio=val_ratio)
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    sc_X = MinMaxScaler()
    train_dataset = sc_X.fit_transform(train_dataset)
    val_dataset = sc_X.transform(val_dataset)
    test_dataset = sc_X.transform(test_dataset)
    logger.info('Datasets sizes: {}, {}, {}'.format(len(train_idx),
                                                    len(val_idx),
                                                    len(test_idx)))
    model_manager = ModelManager()
    model = model_manager.build(ModelManager.MODEL_TYPE_LOG_PARAMS,
                                input_size=window_size,
                                lstm_units=LSTM_units,
                                num_params=num_params)
    X_train, y_train = data_preprocess.generate(train_dataset, window_size)
    X_val, y_val = data_preprocess.generate(val_dataset, window_size)
    X_test, y_test = data_preprocess.generate(test_dataset, window_size)
    model_trainer = ModelTrainer(logger, epochs=max_epochs,
                                 early_stop=early_stop,
                                 batch_size=batch_size)
    # Run training and validation to fit the model
    model_trainer.train(model, [X_train, y_train], [X_val, y_val],
                        metric_index='mse',
                        out_tensorboard_path=out_tensorboard_path)
    model.summary()
    # Save the model
    model_manager.save(model, output_path, 'log_par_model.h5')
    y_val_pred = model.predict(X_val)
    # Model Evaluation:
    model_evaluator = ModelEvaluator(logger)
    interval, mse_val = model_evaluator.build_confidence_intervals(y_val,
                                                                   y_val_pred,
                                                                   alpha)
    model_evaluator.plot_confidence_intervals(interval, mse_val, alpha)
    y_test_pred = model.predict(X_test)
    anomalies_idx, mse_test = model_evaluator.get_anomalies_idx(y_test,
                                                                y_test_pred,
                                                                interval)
    model_evaluator.plot_test_errors(interval, mse_test, alpha)
    model_evaluator.plot_time_series(sc_X, val_idx[window_size - 1:],
                                     y_val, y_val_pred,
                                     label='Forecasting results of validation '
                                           'set: ')
    model_evaluator.plot_time_series(sc_X, test_idx[window_size - 1:],
                                     y_test, y_test_pred,
                                     label='Forecasting result of the testing '
                                           'set: ')
    logger.info("Number of anomalies: {}".format(len(anomalies_idx)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_parameters_model_runner_args(parser)

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    try:
        os.makedirs(args.output_path, exist_ok=True)
    except OSError as error:
        logger.error("Directory {} can not be created".format(args.output_path))
        exit(1)

    run_parameter_detection_model(logger, args.input_file,
                                  args.output_path, args.window_size,
                                  args.LSTM_units, args.max_epochs,
                                  args.train_ratio, args.val_ratio,
                                  args.early_stop, args.batch_size,
                                  args.out_tensorboard_path, args.alpha)
