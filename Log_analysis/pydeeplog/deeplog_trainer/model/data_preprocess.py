import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataPreprocess:
    """
    Description:
    + Encodes log keys with additional values for padding, unknown keys and
    ending key
    + Randomly splits the dataset as usually: train (70%), validation (15%)
    and test (15%)
    + Splits into chunks of size `WINDOW_SIZE`. Note that, the longer the
    window size is, the more accurate the model is. However, workflows are
    less accurate.
    """
    # In data_preprocess.py

class DataPreprocess:
    def __init__(self, vocab: list = None, dict_token2idx: dict = None, dict_idx2token: list = None):
        """
        Attributes:
        :param vocab (list): List of unique keys in the dataset. Used if dict_token2idx/dict_idx2token are not provided.
        :param dict_token2idx (dict): Precomputed token to index mapping.
        :param dict_idx2token (list): Precomputed index to token mapping.
        """
        self.special_tokens = ['[PAD]', '[UNK]', '[END]']

        if dict_token2idx is not None and dict_idx2token is not None:
            self.dict_token2idx = dict_token2idx
            self.dict_idx2token = dict_idx2token
            self.num_tokens = len(self.dict_idx2token)
            # PAD token must be in the position 0
            if '[PAD]' not in self.dict_token2idx or self.dict_token2idx['[PAD]'] != 0:
                # This might happen if loaded dicts are malformed or don't include special tokens correctly
                # For robustness, you might want to re-ensure special tokens if they are missing,
                # but ideally, the saved dicts should be complete.
                # For now, we'll assume they are correctly saved.
                # If not, one would need to rebuild them carefully.
                # Example: If special tokens were NOT part of the saved dicts.
                # self.dict_idx2token = self.special_tokens + [t for t in dict_idx2token if t not in self.special_tokens]
                # self.dict_token2idx = {token: idx for idx, token in enumerate(self.dict_idx2token)}
                # self.num_tokens = len(self.dict_idx2token)
                pass # Assuming saved dicts are correct and include special tokens.
            assert self.dict_token2idx.get('[PAD]') == 0, \
                   f"[PAD] token not at index 0 or missing. Found: {self.dict_token2idx.get('[PAD]')}"
            # Reconstruct vocab from dict_idx2token if needed for other methods, excluding special tokens
            self.vocab = [token for token in self.dict_idx2token if token not in self.special_tokens]

        elif vocab is not None:
            self.vocab = vocab
            self.num_tokens = len(self.vocab) + len(self.special_tokens)
            # Build dictionaries of tokens
            self.dict_idx2token = self.special_tokens + self.vocab # self.vocab should be unique keys
            self.dict_token2idx = {value: key for key, value in
                                   enumerate(self.dict_idx2token)}
            # PAD token must be in the position 0
            assert self.dict_token2idx['[PAD]'] == 0
        else:
            raise ValueError("Either (vocab) or (dict_token2idx and dict_idx2token) must be provided.")

    # ... (rest of the DataPreprocess class remains the same) ...
    # Ensure get_num_tokens, encode_dataset, chunks, chunks_seq, transform, split_idx, get_dictionaries are there

    def get_num_tokens(self):
        """
        Returns number of log keys + the number of special tokens
        """
        return self.num_tokens

    def encode_dataset(self, dataset):
        """
        Encodes values of the dataset. The unknown tokens are replaced by the
        the corresponding index of the token 'UNK'.
        """
        for i, seq in enumerate(dataset):
            dataset[i] = [self.dict_token2idx[x] if x in self.dict_token2idx
                          else self.dict_token2idx['[UNK]'] for x in seq]
        return dataset

    def chunks(self, dataset, window_size):
        """
        Splits the dataset in smaller chunks with window_size as maximum length.
        """
        chunks = []
        for seq in dataset:
            chunks += self.chunks_seq(seq, window_size)
        return chunks

    @staticmethod
    def chunks_seq(seq, window_size):
        """
        Splits a sequence in smaller chunks with window_size as maximum length.
        Return a list of lists of size (len(seq)-window_size+1,window_size)
        """
        chunks = []
        # If the sequence is longer than the window size, drag the window
        # and split as many sequences as possible
        if len(seq) > window_size:
            i = 0
            while i + window_size <= len(seq):
                x = seq[i:(i + window_size)]
                chunks.append(x)
                i += 1
        else:
            chunks = [seq]
        return chunks

    def transform(self, dataset, add_padding=0):
        """
        Prepares the data to be consumed by the ML model. If used, it should
        come after chunks method.
        """
        # Split into input and target values
        X_data = []
        y_data = []
        for seq in dataset:
            X_data.append(seq[:-1] + [self.dict_token2idx['[END]']])
            y_data.append(seq[-1])

        # Add padding if necessary
        if add_padding > 0:
            # Pads input sequences. The ones whose length is smaller than
            # 'maxlen' padded with 'value' until they reach all the same length
            X_data = pad_sequences(X_data, maxlen=add_padding,
                                   value=self.dict_token2idx['[PAD]'],
                                   padding='post')
        # One hot encoding: Return vectors with all zeros but 1 in the in the
        # indices position passed in input.
        X_data = np.array(tf.one_hot(X_data, self.num_tokens))
        y_data = np.array(tf.one_hot(y_data, self.num_tokens))

        return X_data, y_data

    @staticmethod
    def split_idx(dataset_size, train_ratio, val_ratio):
        """
        Description: splits indices of the data into the usual three subsets:
        training, validation and testing.
        :param: dataset_size: length of dataset
        :param: train_ratio (float): defines the subset for training.
        :param: val_ratio (float): defines the subset for validation
        (must be greater than train_ratio).
        :return three subsets of the input dataset.
        """
        train_idx, val_idx, test_idx = \
            np.split(np.arange(dataset_size),
                     [int(train_ratio * dataset_size),
                      int(val_ratio * dataset_size)])
        return train_idx, val_idx, test_idx

    def get_dictionaries(self):
        return self.dict_idx2token, self.dict_token2idx
