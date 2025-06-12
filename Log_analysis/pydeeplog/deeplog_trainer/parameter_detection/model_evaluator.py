from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


class ModelEvaluator:
    """
    Description: To evaluate the parameter detection anomaly detection model,
    Deeplog calculates the MSE between each vector in validation set and
    its prediction to generate a Gaussian distribution. At deployment, if the
    error between a prediction and an observed value vector is within a
    high-level of confidence interval of the above Gaussian distribution,
    the parameter value vector of the incoming log entry is considered normal,
    otherwise abnormal.
    """

    def __init__(self, logger):
        """
        :param logger: logger function from logging module
        """
        self.logger = logger

    def plot_time_series(self, scaler, timestamps, y_true, y_pred, label):
        """
        Description: Plot time series trend for each parameter as well as its
        predictions
        :param scaler: Object of the package sklearn used to normalize the
        dataset
        :param timestamps: original idxs of the data to be plotted
        :param y_true: true value vectors to be predicted
        :param y_pred: model predictions of the `y_true` array
        :param label: title of the graph
        """
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)
        num_params = y_true.shape[1]
        for i in range(1, num_params):
            plt.figure(figsize=(20, 6))
            plt.plot(timestamps, y_true[:, i], 'o-', mfc='none',
                     label='Real')
            plt.plot(timestamps, y_pred[:, i], 'o-',
                     mfc='none', label='Prediction', color='red')
            plt.xticks(timestamps)
            plt.title(label + 'Parameter' + str(i))
            plt.legend()
            plt.show()

    def get_mses(self, y_data: list, y_pred: list):
        """
        :param y_data: outputs array to predict
        :param y_pred: predictions of `y_data`
        :return: Mean squared errors array between `y_data` and `y_pred`
        """
        errors = []
        for i in range(len(y_data)):
            errors.append(mean_squared_error(y_data[i], y_pred[i]))
        return errors

    def build_confidence_intervals(self, y_val: list, y_val_pred: list,
                                   alpha: float = 0.95):
        """
        :param y_val: outputs array of the validation set
        :param y_val_pred: predictions of the outputs of the validation set
        :param alpha: confidence level of the confidence interval
        :return: the two endpoints of the confidence interval of the Gaussian
        distribution of MSEs from the validation set, as well as the MSE values
        """
        mse_val = self.get_mses(y_val, y_val_pred)
        mu = np.mean(mse_val)
        sd = np.std(mse_val)
        # Model the MSEs as a Gaussian distribution and build the confidence
        # interval of level alpha
        interval = stats.norm.interval(alpha, mu, sd)
        return interval, mse_val

    def plot_confidence_intervals(self, interval, mse_val, alpha=0.95):
        """
        Description: Plot the Gaussian distribution and the related confidence
        interval.
        :param interval: Confidence interval of the Gaussian distribution built
        with the MSEs from the validation set
        :param mse_val: MSEs between each vector in the validation set and its
        prediction
        :param alpha: confidence level of the confidence interval
        """
        mu = np.mean(mse_val)
        sd = np.std(mse_val)
        x = np.linspace(0, max(mse_val), 1000)
        pdf = stats.norm.pdf(x, mu, sd)
        plt.plot(x, pdf)
        plt.fill_between(x, pdf, where=(x > interval[0]) & (x <= interval[1]),
                         color='blue')
        plt.title('Confidence interval of level {} for validation errors'
                  .format(alpha))
        plt.show()

    def get_anomalies_idx(self, y_true: list, y_pred: list,
                          interval: list):
        """
        :param y_true: outputs array of the testing set
        :param y_pred: predictions of the outputs of the testing set
        :param interval: confidence interval of the Gaussian distribution built
        with the MSEs from the validation set
        :return: indexes of data labeled as anomalous as well as a graph showing
        the trend of the MSEs.
        """
        mse_test = self.get_mses(y_true, y_pred)
        anomalies_idx = [i for i in range(len(mse_test))
                         if mse_test[i] > interval[1]
                         or mse_test[i] < interval[0]]
        return anomalies_idx, mse_test

    def plot_test_errors(self, interval, mse_test, alpha=0.95):
        """
        :param interval: confidence interval of the Gaussian distribution built
        with the MSEs from the validation set
        :param alpha: confidence level of the confidence interval
        :param mse_test: MSEs between each vector in the testing set and its
        prediction
        """
        plt.hlines(interval[1], 0, len(mse_test), colors='red', linestyles='--',
                   label='CI={:d}%'.format(int(alpha * 100)))
        plt.legend()
        plt.title('Value vector errors')
        plt.ylabel('MSE')
        plt.plot(mse_test)
        plt.show()
