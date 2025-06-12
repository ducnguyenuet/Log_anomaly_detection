import argparse
import ast

import configparser
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from deeplog_trainer.log_parser.adapter import AdapterFactory, ParseMethods
from deeplog_trainer.log_parser.drain import Drain
from deeplog_trainer.log_parser.sessions import SessionStorage

from . import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')


def run_drain(logger, input_file, output_path, config_file, window_size):
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config = TemplateMinerConfig()
    config.load(config_file)
    template_miner = TemplateMiner(config=config)
    adapter_factory = AdapterFactory()
    parser = configparser.ConfigParser()
    parser.read(os.path.join(config_file))
    adapter_params = dict(parser['ADAPTER_PARAMS'])
    adapter_params.setdefault('anomaly_labels', '[]')
    adapter_params['anomaly_labels'] = ast.literal_eval(
        adapter_params['anomaly_labels'])

    if 'regex' in parser.options('ADAPTER_PARAMS'):
        adapter_params['regex'] = ast.literal_eval(adapter_params['regex'])
    elif 'delta' in parser.options('ADAPTER_PARAMS'):
        adapter_params['delta'] = ast.literal_eval(adapter_params['delta'])

    adapter = adapter_factory.build_adapter(**adapter_params)
    drain = Drain(template_miner)
    session_storage = SessionStorage()
    logger.info(f"Drain3 started reading from {input_file}")
    line_count = 0
    logformat = parser.get('ADAPTER_PARAMS', 'logformat')
    headers, text_regex = ParseMethods.generate_logformat_regex(
        logformat=logformat)
    with open(os.path.join(root_path, input_file), 'r') as f:
        for line in f:
            sess_id, anomaly_flag = adapter.get_session_id(log=line)
            match = text_regex.search(line.strip())
            message = match.group('Content')
            drain_result = drain.add_message(message)
            sessions = session_storage.get_sessions(sess_id,
                                                    drain_result['template_id'])
            templates = session_storage.get_templates(
                drain_result['template_id'], drain_result['template'])
            parameters = session_storage.get_parameters(sess_id,
                                                        drain_result['params'])
            line_count += 1

            if line_count % 1000 == 0:
                logger.info('Processed {} log lines.'.format(line_count))

        logger.info(f'Finished Drain3 process. Total of lines: {line_count}')
    # Import the results in json format
    result = {'data': []}

    if anomaly_flag:
        for sess_id in sessions:
            result['data'].append(dict(template_seq=sessions[sess_id],
                                       template_params=parameters[sess_id],
                                       is_abnormal=anomaly_flag[sess_id],
                                       session_id=sess_id))
    else:
        for sess_id in sessions:
            result['data'].append(dict(template_seq=sessions[sess_id],
                                       template_params=parameters[sess_id],
                                       session_id=sess_id))
    with open(os.path.join(output_path, 'data.json'), 'w') as f:
        json.dump(result, f)
    # Save the Drain tree object in a JSON file
    with open(os.path.join(output_path, 'drain.json'), 'w') as h:
        drain_serialized = drain.serialize_drain()
        json.dump(drain_serialized, h)
    with open(os.path.join(output_path, 'session_grouper_conf.json'),
              'w') as sg:
        sg_dict = {'session_grouper_type': "RegexSessionGrouper",
                   'text_regex': text_regex.pattern.replace('?P', '?'),
                   'session_id_regex': adapter_params.setdefault('regex', ''),
                   'session_delimiters': [adapter_params['delimiter']] if
                   'delimiter' in adapter_params else [],
                   'window_size': window_size
                   }
        json.dump(sg_dict, sg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_drain_runner_args(parser)

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    try:
        os.makedirs(args.output_path, exist_ok=True)
    except OSError as error:
        logger.error("Directory {} can not be created".format(args.output_path))
        exit(1)
    run_drain(logger, args.input_file, args.output_path, args.config_file,
              args.window_size)