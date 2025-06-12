import re
from datetime import datetime
from datetime import timedelta
from abc import ABCMeta, abstractmethod


class SessionAdapterInterface(metaclass=ABCMeta):
    """Adapter Sessions Interface"""

    @abstractmethod
    def get_session_id(self, log: str):
        """
        The interface method: given the log message,
        returns the corresponding session Id it belongs to.
        """
        pass


class OnlyRegex(SessionAdapterInterface):
    """
    This Class implements the SessionAdapter interface in the case only an
    identifier in regex format is provided in order to group the logs in
    different sessions
    """

    def __init__(self, regex, anomaly_labels):
        """
        :param regex: The session identifier in format “r-string”. For example,
        it could be the id of a particular process.
        :param anomaly_labels: List with the strings leading to an anomaly
        """
        self.regex = regex
        self.anomaly_labels = anomaly_labels
        self.d = {}
        self.last_session_id = 0
        self.anomaly_flag = {}

    def get_session_id(self, log: str):
        log = log.strip()
        identifier = re.search(self.regex, log)[0]
        if identifier not in self.d:
            self.d[identifier] = self.last_session_id + 1
            self.last_session_id += 1
            if self.anomaly_labels:
                self.anomaly_flag[self.d[identifier]] = False
        if self.anomaly_labels and not self.anomaly_flag[self.d[identifier]]:
            self.anomaly_flag[self.d[identifier]] = any(anomaly_label in log
                                                        for anomaly_label in
                                                        self.anomaly_labels)
        return self.d[identifier], self.anomaly_flag


class RegexAndDelimiter(SessionAdapterInterface):
    """
    This Class implements the SessionAdapter interface in the case, in order to
    group the logs in different sessions, both an identifier in regex format and
    a delimiter string are provided.
    """

    def __init__(self, regex, delimiter: str, anomaly_labels):
        """
        :param regex: The identifier in format “r-string”. For example,
        it could be the id of a particular process.
        :param delimiter: The string sentence that calls a new session.
        :param anomaly_labels:  List with the strings leading to an anomaly
        """
        self.regex = regex
        self.delimiter = delimiter
        self.d = {}
        self.last_session_id = 0
        self.anomaly_flag = {}
        self.anomaly_labels = anomaly_labels

    def get_session_id(self, log: str):
        log = log.strip()
        identifier = re.search(self.regex, log)[0]
        if identifier not in self.d:
            self.d[identifier] = self.last_session_id + 1
            if self.anomaly_labels:
                self.anomaly_flag[self.d[identifier]] = False
        if self.delimiter in log:
            self.d[identifier] = self.last_session_id + 1
            self.last_session_id += 1
            if self.anomaly_labels:
                self.anomaly_flag[self.d[identifier]] = False
        if self.anomaly_labels and not self.anomaly_flag[self.d[identifier]]:
            self.anomaly_flag[self.d[identifier]] = any(anomaly_label in log
                                                        for anomaly_label in
                                                        self.anomaly_labels)
        return self.d[identifier], self.anomaly_flag


class OnlyDelimiter(SessionAdapterInterface):
    """
    This Class implements the SessionAdapter interface in the case,
    in order to group the logs in different sessions, only a delimiter string is
    provided
    """

    def __init__(self, delimiter: str, anomaly_labels):
        """
        :param delimiter: The string sentence that calls a new session.
        :param anomaly_labels: List with the strings leading to an anomaly
        """
        self.last_session_id = 0
        self.anomaly_flag = {}
        self.delimiter = delimiter
        self.anomaly_labels = anomaly_labels

    def get_session_id(self, log: str):
        log = log.strip()
        if self.delimiter in log:
            self.last_session_id += 1
            if self.anomaly_labels:
                self.anomaly_flag[self.last_session_id] = False
        if self.anomaly_labels and not self.anomaly_flag[self.last_session_id]:
            self.anomaly_flag[self.last_session_id] = any(anomaly_label in log
                                                          for anomaly_label in
                                                          self.anomaly_labels)
        return self.last_session_id, self.anomaly_flag


class TimeInterval(SessionAdapterInterface):
    """
    This Class implements the SessionAdapter interface in the case, in order to
    group the logs in different sessions, the system entry time of each log is
    provided. In particular, a new session is created every time the time
    elapsed is bigger of a fixed time interval (provided by the user as well).
    """

    def __init__(self, logformat, time_format: str, delta: dict,
                 anomaly_labels):
        """
        :param logformat: Format of the entry log. The variable must contain the
        word 'Time' to calculate the time interval between the entry log and
        another one.
        :param time_format: Set the format of time. Example: '%H:%M:%S.%f'.
        :param delta: Dictionary indicating the time that must elapse to create
        a new session. Example: {'minutes'=1, 'seconds'=30}
        :param anomaly_labels: List with the strings leading to an anomaly.
        """
        self.logformat = logformat
        if 'Time' not in self.logformat:
            raise Exception('The logformat must contain the word `Time` in'
                            'order to identify the entry time of the logs')
        self.last_session_id = 0
        self.anomaly_flag = {}
        self.delta = delta
        self.time_format = time_format
        self.starter_time = 0
        self.anomaly_labels = anomaly_labels

    def get_session_id(self, log: str):
        delta = timedelta(**self.delta)
        headers, regex = ParseMethods.generate_logformat_regex(self.logformat)
        match = regex.search(log.strip())
        time = match.group('Time')
        if not self.starter_time:
            self.starter_time = time
            self.last_session_id = 1
        elapsed_time = \
            datetime.strptime(time, self.time_format) - datetime.strptime(
                self.starter_time, self.time_format)
        if elapsed_time >= delta:
            self.last_session_id += 1
            self.starter_time = time
            if self.anomaly_labels:
                self.anomaly_flag[self.last_session_id] = False
        if self.anomaly_labels and not self.anomaly_flag[
            self.last_session_id
        ]:
            self.anomaly_flag[self.last_session_id] = \
                any(anomaly_label in log for anomaly_label in
                    self.anomaly_labels)
        return self.last_session_id, self.anomaly_flag


class ParseMethods:

    def __init__(self):
        pass

    @staticmethod
    def generate_logformat_regex(logformat):
        """
        This method allows to split the log messages in different parts. For
        example, it is fundamental to isolate only the content part of the log
        to send to Drain.
        In particular, given the format of the log, for example
        '<Pid> <Content>', it outputs the headers (['Pid', 'Content']), and the
        relative regex generated.
        """
        # 'Content' word must be in logformat variable to isolate the log part
        # to be parsed by Drain
        if 'Content' not in logformat:
            raise Exception('The logformat must contain the word `Content` in'
                            'order to isolate the log part to be parsed by '
                            'Drain')
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex


class AdapterFactory:
    """The factory class"""
    ADAPTER_TYPE_DELIMITER = 'only_delimiter'
    ADAPTER_TYPE_DELIMITER_AND_REGEX = 'delimiter+regex'
    ADAPTER_TYPE_REGEX = 'only_regex'
    ADAPTER_TYPE_INTERVAL_TIME = 'interval_time'

    def build_adapter(self, adapter_type: str, **kwargs):
        if adapter_type == AdapterFactory.ADAPTER_TYPE_DELIMITER:
            self._validate_delimiter_kwargs(kwargs)
            return OnlyDelimiter(delimiter=kwargs['delimiter'],
                                 anomaly_labels=kwargs['anomaly_labels'])
        if adapter_type == AdapterFactory.ADAPTER_TYPE_REGEX:
            self._validate_regex_kwargs(kwargs)
            return OnlyRegex(regex=kwargs['regex'],
                             anomaly_labels=kwargs['anomaly_labels'])
        if adapter_type == AdapterFactory.ADAPTER_TYPE_DELIMITER_AND_REGEX:
            self._validate_regex_and_delimiter_kwargs(kwargs)
            return RegexAndDelimiter(regex=kwargs['regex'],
                                     delimiter=kwargs['delimiter'],
                                     anomaly_labels=kwargs[
                                         'anomaly_labels'])
        if adapter_type == AdapterFactory.ADAPTER_TYPE_INTERVAL_TIME:
            self._validate_time_interval_kwargs(kwargs)
            return TimeInterval(logformat=kwargs['logformat'],
                                time_format=kwargs['time_format'],
                                delta=kwargs['delta'],
                                anomaly_labels=kwargs['anomaly_labels'])
        raise Exception('Adapter type not found')

    def _validate_delimiter_kwargs(self, kwargs):
        if not {'delimiter', 'anomaly_labels'}.issubset(set(kwargs)):
            raise ValueError('Provide right parameters')

    def _validate_regex_kwargs(self, kwargs):
        if not {'regex', 'anomaly_labels'}.issubset(set(kwargs)):
            raise ValueError('Provide right parameters')

    def _validate_regex_and_delimiter_kwargs(self, kwargs):
        if not {'delimiter', 'regex', 'anomaly_labels'}.issubset(set(kwargs)):
            raise ValueError('Provide right parameters')

    def _validate_time_interval_kwargs(self, kwargs):
        if not {'logformat', 'time_format', 'delta', 'anomaly_labels'
                }.issubset(set(kwargs)):
            raise ValueError('Provide right parameters')
