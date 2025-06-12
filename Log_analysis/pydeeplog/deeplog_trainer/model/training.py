import tensorflow as tf
import numpy as np


class ValLossLogger(tf.keras.callbacks.Callback):
    def __init__(self, logger, loss_index: str, metric_index: str):
        """
        Attributes
        :param logger: logger function from logging module
        :param loss_index: loss function name
        :param metric_index: metric index
        """
        self.logger = logger
        self.loss_index = loss_index
        self.metric_index = metric_index
        # build dictionary to save the best loss and metric values during the
        # training and the validation process
        self.best_loss = {'train': np.Inf, 'val': np.Inf}
        self.best_metric = {'train': 0, 'val': 0}
        # Declare that the ValLossLogger class inherits from the Callback
        # class in Keras
        super().__init__()

    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        self.logger.info('Start training')

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch printing out the corresponding
        loss/accuracy values
        """
        if logs['val_' + self.metric_index] > self.best_metric['val']:
            self.best_loss['train'] = logs[self.loss_index]
            self.best_loss['val'] = logs['val_' + self.loss_index]
            self.best_metric['train'] = logs[self.metric_index]
            self.best_metric['val'] = logs['val_' + self.metric_index]

        msg = 'Epoch: %d - %s: %.4f (%s: %.4f) - Val. %s:  %.4f (%s: %.4f)' \
              % (epoch, self.loss_index, logs[self.loss_index],
                 self.metric_index, logs[self.metric_index],
                 self.loss_index, logs['val_' + self.loss_index],
                 self.metric_index, logs['val_' + self.metric_index])
        self.logger.info(msg)

    def on_train_end(self, logs=None):
        """
        Called at the end of the training process. The loss/accuracy value
        of the best model are displayed
        """
        self.logger.info('%s: %.4f (%s: %.4f) - Val. %s: %.4f '
                         '(%s: %.4f)' % (self.loss_index,
                                         self.best_loss['train'],
                                         self.metric_index,
                                         self.best_metric['train'],
                                         self.loss_index,
                                         self.best_loss['val'],
                                         self.metric_index,
                                         self.best_metric['val']))
        self.logger.info('Training finished')


class ModelTrainer:
    """
    Description: Once the numerical representation of each session is ready,
    DeepLog treats these sequences as a Multi-class Time Series
    Classification, which is a supervised learning problem aiming to predict
    class labels over the time based on past behaviour.
    """
    def __init__(self, logger, epochs: int, batch_size: int, early_stop: int):
        """
        :param logger: logger function from logging module
        :param epochs: Number of times the entire dataset is passed forward and
         backward by the neural network
        :param batch_size: Number of samples that will be propagated through the
         network.
        :param early_stop: Number of epochs with no improvement after which
         training will be stopped.
        """
        self.logger = logger
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop = early_stop

    def train(self, model, train_dataset, val_dataset,
              out_tensorboard_path=None, loss_index='loss',
              metric_index='accuracy'):
        """
        Given the model, the train and validation set as arguments, the method
        train process is trigger. The model checkpoints allows to save the model
        according to the 'monitor' index.
        :param model: a Keras model instance
        :param train_dataset: train set type array
        :param val_dataset: validation set type array
        :param out_tensorboard_path: output path where to save the
        tensorboard results
        :param loss_index: loss function name
        :param metric_index: metric index
        """
        train_logger = ValLossLogger(self.logger, loss_index, metric_index)
        early_stop = tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor='val_loss',
            patience=self.early_stop,
            restore_best_weights=True,
            verbose=0)
        callbacks = [train_logger, early_stop]
        if out_tensorboard_path is not None:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=out_tensorboard_path,
                histogram_freq=0,
                embeddings_freq=0)
            callbacks.append(tensorboard_callback)
        history = model.fit(
                train_dataset[0],
                train_dataset[1],
                validation_data=(val_dataset[0], val_dataset[1]),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                shuffle=False,
                verbose=0
        )
        return history
