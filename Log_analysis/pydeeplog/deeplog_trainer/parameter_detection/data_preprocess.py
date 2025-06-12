import numpy as np

class DataPreprocess:
    def generate(self, dataset, window_size):
        """
        Splits the dataset in smaller chunks with window_size as maximum length.
        """
        n_items = len(dataset)
        n_params = np.shape(dataset)[1]
        inputs = []
        outputs = []
        assert window_size < n_items
        i = 0
        while i + window_size <= n_items:
            x = dataset[i:i+window_size-1]
            x = np.vstack((x, np.zeros(n_params)))
            x[-1][0] = dataset[i+window_size-1][0]
            y = dataset[i+window_size-1]
            inputs.append(x)
            outputs.append(y)
            i += 1
        return np.array(inputs), np.array(outputs)

    @staticmethod
    def split_idx(dataset_size, train_ratio, val_ratio):
        """
        Description: splits indices of the data into the usual three subsets:
        training, validation and testing.
        :param: dataset_size: length of dataset
        :param: train_ratio (float): defines the subset for training.
        :param: val_ratio (float): defines the subset for validation (must be greater
        than train_ratio).
        :return three subsets of the input dataset.
        """
        train_idx, val_idx, test_idx = \
            np.split(np.arange(dataset_size),
                     [int(train_ratio * dataset_size),
                      int(val_ratio * dataset_size)])
        return train_idx, val_idx, test_idx
