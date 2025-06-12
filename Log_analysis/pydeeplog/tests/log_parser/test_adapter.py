from deeplog_trainer.log_parser.adapter import AdapterFactory
import pytest


@pytest.fixture(scope='session')
def batrasio_adapter():
    adapter_factory = AdapterFactory()
    return adapter_factory.build_adapter(
        adapter_type=AdapterFactory.ADAPTER_TYPE_DELIMITER_AND_REGEX,
        delimiter='TCP source connection created',
        anomaly_labels=['TCP source SSL error', 'TCP source socket error'],
        regex=r"^(\d+)")

def get_batrasio_data():
    expected_sess_id = [1, 1, 1, 2, 2, 2, 2, 3, 2, 3, 3, 3, 3]
    expected_anomaly_flag = [False, False, False, False, False, True, True,
                             False, True, False, False, False, False]
    with open('data/sample_test_batrasio.log') as f:
        for i, line in enumerate(f):
            yield line.strip(), expected_sess_id[i], expected_anomaly_flag[i]

@pytest.mark.parametrize("logs, expected_sess_id, expected_anomaly_flag",
                         get_batrasio_data())
def test_batrasio_sessions(logs, expected_sess_id, expected_anomaly_flag,
                           batrasio_adapter):
    sess_id, anomaly_flag = batrasio_adapter.get_session_id(log=logs)
    assert sess_id == expected_sess_id
    # The flag indicator of the anomalous nature of the sessions must be a
    # boolean
    assert anomaly_flag[sess_id] == expected_anomaly_flag


@pytest.fixture(scope='session')
def hdfs_adapter():
    adapter_factory = AdapterFactory()
    return adapter_factory.build_adapter(
        adapter_type=AdapterFactory.ADAPTER_TYPE_REGEX,
        regex=r'blk_-?\d+',
        anomaly_labels=[])

def get_hdfs_data():
    expected_sess_id = [1, 1, 2, 2, 2, 1, 1, 2, 3, 4, 1, 4, 3, 3, 3]
    expected_block_ids = {'blk_-1608999687919862906',
                          'blk_7503483334202473044',
                          'blk_-3544583377289625738',
                          'blk_-9073992586687739851'}
    with open('data/sample_test_hdfs.log') as f:
        for i, line in enumerate(f):
            yield line.strip(), expected_sess_id[i], expected_block_ids

@pytest.mark.parametrize("logs, expected_sess_id, expected_block_ids",
                         get_hdfs_data())
def test_hdfs_sessions(logs, expected_sess_id, expected_block_ids,
                       hdfs_adapter):
    sess_id, anomaly_flag = hdfs_adapter.get_session_id(log=logs)
    assert sess_id == expected_sess_id
    assert set(hdfs_adapter.d.keys()).issubset(expected_block_ids)


@pytest.fixture(scope='session')
def only_delimiter_adapter():
    adapter_factory = AdapterFactory()
    return adapter_factory.build_adapter(
        adapter_type=AdapterFactory.ADAPTER_TYPE_DELIMITER,
        delimiter='TCP source connection created',
        anomaly_labels=['TCP source SSL error', 'TCP source socket error'])

def get_no_procid_data():
    expected_sess_id = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    expected_anomaly_flag = [False, False, False, False, False, False, False,
                             False, False, False, False, False, False]
    with open('data/sample_test_no_procid.log') as f:
        for i, line in enumerate(f):
            yield line.strip(), expected_sess_id[i], expected_anomaly_flag[i]

@pytest.mark.parametrize("logs, expected_sess_id, expected_anomaly_flag",
                         get_no_procid_data())
def test_no_procid_sessions(logs, expected_sess_id, expected_anomaly_flag,
                            only_delimiter_adapter):
    sess_id, anomaly_flag = only_delimiter_adapter.get_session_id(log=logs)
    assert sess_id == expected_sess_id
    assert anomaly_flag[sess_id] == expected_anomaly_flag


@pytest.fixture(scope='session')
def box_unix_adapter():
    adapter_factory = AdapterFactory()
    return adapter_factory.build_adapter(
        adapter_type=AdapterFactory.ADAPTER_TYPE_INTERVAL_TIME,
        logformat='<Date> <Time>, <Content>',
        time_format='%H:%M:%S.%f',
        delta={'milliseconds': 2},
        anomaly_labels=[])

def get_box_unix_data():
    expected_sess_id = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 5, 5, 6]
    with open('data/sample_test_box_unix.log') as f:
        for i, line in enumerate(f):
            yield line.strip(), expected_sess_id[i]

@pytest.mark.parametrize("logs, expected_sess_id", get_box_unix_data())
def test_box_unix_sessions(logs, expected_sess_id, box_unix_adapter):
    sess_id, anomaly_flag = box_unix_adapter.get_session_id(log=logs)
    assert sess_id == expected_sess_id


@pytest.mark.parametrize("adapter_type, kwargs",
                         [(pytest.param(
                             'unknown adapter',
                             {'delimiter': 'TCP source connection created',
                              'anomaly_labels': ['TCP source SSL error',
                                                 'TCP source socket error']},
                             marks=pytest.mark.xfail(raises=Exception))),
                          (pytest.param(
                              AdapterFactory.ADAPTER_TYPE_DELIMITER_AND_REGEX,
                              {'delimiter': 'TCP source connection created',
                               'anomaly_labels': []},
                              marks=pytest.mark.xfail(raises=ValueError))),
                          (pytest.param(
                              AdapterFactory.ADAPTER_TYPE_INTERVAL_TIME,
                              {'logformat': '<Date>, <Content>',
                               'time_format': '%H:%M:%S.%f',
                               'delta': {'milliseconds': 2},
                               'anomaly_labels': []},
                              marks=pytest.mark.xfail(raises=Exception)))])
def test_exceptions(adapter_type, kwargs):
    adapter_factory = AdapterFactory()
    adapter_factory.build_adapter(adapter_type, **kwargs)
