from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from deeplog_trainer.log_parser.drain import Drain
from deeplog_trainer.log_parser.adapter import ParseMethods
import pytest
import re
from deeplog_trainer import SERIAL_DRAIN_VERSION


@pytest.fixture()
def drain():
    config = TemplateMinerConfig()
    config.load('tests/test_drain.ini')
    template_miner = TemplateMiner(config=config)
    return Drain(template_miner)


def get_log():
    with open('data/sample_test_batrasio.log') as f:
        for line in f:
            yield line.strip()


def get_logs():
    data = []
    with open('data/sample_test_batrasio.log') as f:
        for line in f:
            data.append(line)
        return data[0:3]


def get_expected_result():
    result = {'version': SERIAL_DRAIN_VERSION,
              'depth': 4,
              'similarityThreshold': 0.5,
              'maxChildrenPerNode': 100,
              'delimiters': [' '],
              'masking': [{"regex_pattern":
                               "((?<=[^A-Za-z0-9])|^)(([0-9a-f]{2,}:)"
                               "{3,}([0-9a-f]{2,}))((?=[^A-Za-z0-9])|$)",
                           "mask_with": "<ID>"},
                          {"regex_pattern":
                               "((?<=[^A-Za-z0-9])|^)(\\d{1,3}\\.\\"
                               "d{1,3}\\.\\d{1,3}\\.\\d{1,3})"
                               "((?=[^A-Za-z0-9])|$)",
                           "mask_with": "<IP>"}]}
    no_msg_result = {**result, 'root': {"depth": 0, "key": "root",
                                        'children': {},
                                        'clusters': []}}
    new_root = {"depth": 0,
                "key": "root",
                "children": {
                    "7": {"depth": 1, "key": "7",
                          "children": {
                              "server=<IP>:": {
                                  "depth": 2,
                                  "key": "server=<IP>:",
                                  "children": {},
                                  "clusters":
                                      [{"clusterId": 1,
                                        "logTemplateTokens":
                                            ["server=<IP>:",
                                             "connector=null,",
                                             "client=<IP>:44940:",
                                             "TCP",
                                             "source",
                                             "connection",
                                             "created"]}]}},
                          "clusters": []},
                    "6": {"depth": 1, "key": "6",
                          "children": {
                              "server=<IP>:": {
                                  "depth": 2,
                                  "key": "server=<IP>:",
                                  "children": {},
                                  "clusters":
                                      [{"clusterId": 2,
                                        "logTemplateTokens":
                                            ["server=<IP>:",
                                             "TLS", "source",
                                             "connection",
                                             "from",
                                             "<IP>:44940"]}]}},
                          "clusters": []},
                    "4": {"depth": 1, "key": "4",
                          "children": {
                              "Multi": {
                                  "depth": 2,
                                  "key": "Multi",
                                  "children": {},
                                  "clusters":
                                      [{"clusterId": 3,
                                        "logTemplateTokens":
                                            ["Multi", "target",
                                             "connection",
                                             "created"]}]}},
                          "clusters": []}},
                "clusters": []}
    with_msg_result = {**result, 'root': new_root}
    return no_msg_result, with_msg_result


@pytest.mark.parametrize("data, no_msg_result, with_msg_result, logformat",
                         [(get_logs(), get_expected_result()[0],
                           get_expected_result()[1],
                           '<Pid>  <Content>')])
def test_serialize_drain(data, no_msg_result, with_msg_result, logformat,
                         drain):
    headers, regex = ParseMethods.generate_logformat_regex(logformat=logformat)
    # Test serializer in 2 cases: when drain is just initialised and when few
    # sample messages are added to it
    assert drain.serialize_drain() == no_msg_result
    for log in data:
        match = regex.search(log.strip())
        message = match.group('Content')
        drain.add_message(message)
    assert drain.serialize_drain() == with_msg_result
