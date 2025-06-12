import argparse
import os
import sys
import json
import logging
import numpy as np
import configparser
import ast
import re
from elasticsearch import Elasticsearch, helpers

from tensorflow.keras.models import load_model

from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from deeplog_trainer.log_parser.drain import Drain
from deeplog_trainer.model.data_preprocess import DataPreprocess
from deeplog_trainer.log_parser.adapter import AdapterFactory, ParseMethods

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

# Regex để parse log
LOG_REGEX = re.compile(
    r'^(?P<timestamp>\d{6} \d{6}) (?P<Thread_ID>\d+) (?P<log_level>\w+) (?P<logger_class>[\w\.\$\-]+): (?P<message>.+)$'
)
BLOCK_ID_REGEX = re.compile(r'(blk_-?\d+)')

def parse_log_line(line):
    match = LOG_REGEX.match(line.strip())
    if not match:
        return None
    log_json = match.groupdict()
    # Tìm block_id trong message
    block_match = BLOCK_ID_REGEX.search(log_json['message'])
    log_json['block_id'] = block_match.group(1) if block_match else None
    log_json['diagnose'] = 'normal'
    return log_json

def load_drain_from_json(drain_json_path):
    with open(drain_json_path, 'r') as f:
        drain_data = json.load(f)
    config = TemplateMinerConfig()
    config.drain_depth = drain_data.get("depth", 4)
    config.drain_similarity_threshold = drain_data.get("similarity_threshold", 0.5)
    config.drain_max_children = drain_data.get("max_children_per_node", 100)
    config.drain_delimiters = drain_data.get("delimiters", [" "])
    config.drain_masking = drain_data.get("masking", [])
    template_miner = TemplateMiner(config=config)
    drain = Drain(template_miner)
    if hasattr(drain, "set_root_from_dict") and "root" in drain_data:
        drain.set_root_from_dict(drain_data["root"])
    return drain

def parse_log_file_by_session(log_file, drain_conf_file, drain):
    """
    Chia log thành các session giống như train, trả về dict {session_id: [template_id, ...]}
    Đồng thời trả về danh sách log_jsons theo thứ tự dòng log.
    """
    parser = configparser.ConfigParser()
    parser.read(drain_conf_file)
    adapter_params = dict(parser['ADAPTER_PARAMS'])
    adapter_params.setdefault('anomaly_labels', '[]')
    adapter_params['anomaly_labels'] = ast.literal_eval(adapter_params['anomaly_labels'])
    if 'regex' in parser.options('ADAPTER_PARAMS'):
        adapter_params['regex'] = ast.literal_eval(adapter_params['regex'])
    elif 'delta' in parser.options('ADAPTER_PARAMS'):
        adapter_params['delta'] = ast.literal_eval(adapter_params['delta'])
    adapter = AdapterFactory().build_adapter(**adapter_params)
    logformat = parser.get('ADAPTER_PARAMS', 'logformat')
    print(f"Using logformat: {logformat}")
    headers, text_regex = ParseMethods.generate_logformat_regex(logformat=logformat)
    print(f"Using headers: {headers}")
    print(f"Using text_regex: {text_regex.pattern}")

    sessions = {}
    log_jsons = []
    log_idx_map = []  # mapping từ index trong sessions về index trong log_jsons
    with open(log_file, 'r') as f:
        for idx, line in enumerate(f):
            # Lấy session_id trước, không bỏ qua dòng nào
            sess_id, _ = adapter.get_session_id(log=line)
            match = text_regex.search(line.strip())
            if not match:
                continue
            message = match.group('Content')
            result = drain.add_message(message)
            template_id = str(result['template_id'])
            sessions.setdefault(sess_id, []).append(template_id)
            # Sau khi đã lấy session và template, mới parse ra json
            log_json = parse_log_line(line)
            if not log_json:
                print('passing line failed:', line.strip())
                continue
            log_jsons.append(log_json)
            log_idx_map.append((sess_id, len(sessions[sess_id])-1, len(log_jsons)-1))
    # print(log_jsons[:3])
    # print(log_idx_map)
    return sessions, log_jsons, log_idx_map

def predict_next_templates(model, template_ids_str_sequence, data_processor,
                           lstm_input_window_size, top_k=1):
    encoded_sequence = [data_processor.dict_token2idx.get(tid, data_processor.dict_token2idx['[UNK]'])
                        for tid in template_ids_str_sequence]
    if len(encoded_sequence) < lstm_input_window_size:
        if len(encoded_sequence) == 0:
            return np.array([]), np.array([])
    windows = data_processor.chunks_seq(encoded_sequence, lstm_input_window_size)
    if not windows:
        return np.array([]), np.array([])
    X_encoded, y_encoded_one_hot = data_processor.transform(windows, add_padding=lstm_input_window_size)
    if X_encoded.shape[0] == 0:
        return np.array([]), np.array([])
    y_true_indices = np.argmax(y_encoded_one_hot, axis=1)
    predicted_probabilities = model.predict(X_encoded)
    top_predicted_indices = np.argsort(predicted_probabilities, axis=1)[:, -top_k:][:, ::-1]
    return top_predicted_indices, y_true_indices

def send_logs_to_elasticsearch(log_jsons, es_host="http://localhost:9200", index_name="logs"):
    # print(log_jsons[:500])
    es = Elasticsearch(es_host)
    actions = [
        {
            "_index": index_name,
            "_source": log_json
        }
        for log_json in log_jsons
    ]
    helpers.bulk(es, actions)

def main(args):
    logger = logging.getLogger(__name__)
    drain = load_drain_from_json(args.drain_json)
    model = load_model(args.model_h5)
    with open(args.config_json, 'r') as f:
        conf = json.load(f)
    window_size = conf['window_size']
    top_k = conf.get('top_candidates', 1)
    if 'dict_token2idx' not in conf or 'dict_idx2token' not in conf:
        logger.error("CRITICAL: Vocabulary mappings (dict_token2idx, dict_idx2token) "
                     "not found in config_json. Please ensure your training process "
                     "saves these into deeplog_conf.json.")
        sys.exit(1)
    dict_token2idx_loaded = conf['dict_token2idx']
    dict_idx2token_loaded = conf['dict_idx2token']
    logger.info(f"Initializing DataPreprocess with loaded vocabulary. "
                f"Number of tokens in dict_idx2token: {len(dict_idx2token_loaded)}")
    data_preprocess = DataPreprocess(dict_token2idx=dict_token2idx_loaded,
                                     dict_idx2token=dict_idx2token_loaded)

    # --- Chia log thành các session giống train ---
    sessions, log_jsons, log_idx_map = parse_log_file_by_session(args.input_log, args.drain_conf, drain)
    logger.info(f"Found {len(sessions)} sessions in log.")

    # for idx, (sess_id, template_ids) in enumerate(sessions.items()):
    #     if idx >= 3:
    #         break
    #     print(f"Session {idx+1}: session_id = {sess_id}")
    #     print(f"  Template IDs: {template_ids}")


    total_preds = []
    total_y_true = []
    pred_log_indices = []  # mapping index của dự đoán về index của log_jsons

    for session_id, template_ids in sessions.items():
        if not template_ids or len(template_ids) < window_size:
            continue  # Bỏ qua session không đủ dài để dự đoán
        if not template_ids:
            continue
        top_preds, y_true = predict_next_templates(model, template_ids, data_preprocess, window_size, top_k)
        if len(top_preds) == 0:
            continue
        total_preds.extend(top_preds)
        total_y_true.extend(y_true)
        # mapping đúng thứ tự index trong session sang log_jsons
        session_log_indices = [None] * len(template_ids)
        for (sess, idx_in_sess, idx_in_log) in log_idx_map:
            if sess == session_id and idx_in_sess < len(session_log_indices):
                session_log_indices[idx_in_sess] = idx_in_log
        pred_log_indices.extend(session_log_indices[window_size-1:])
        # print(len(session_log_indices[window_size-1:]))
        # print(len(y_true))
        

    logger.info(f"định mapping với sô lượng: {len(pred_log_indices)}")
    logger.info(f"số lượng mô hình đã dự đoán được: {len(total_y_true)}")
    logger.info(f"Tổng số dòng: {len(total_preds)}")
    logger.info("Dòng\tTrạng thái\tTemplate thực tế\tTop dự đoán")
    for i in range(len(total_preds)):
        label = "normal" if total_y_true[i] in total_preds[i] else "anomaly"
        logger.info(f"{i}\t{label}\t{total_y_true[i]}\t{total_preds[i]}")
        # Cập nhật trường diagnose trong log_jsons
        if i < len(pred_log_indices):
            log_idx = pred_log_indices[i]
            if label == "anomaly":
                log_jsons[log_idx]['diagnose'] = "anomaly"

    if len(total_preds) > 0:
        num_anomaly = sum(1 for i in range(len(total_preds)) if total_y_true[i] not in total_preds[i])
        anomaly_rate = num_anomaly / len(total_preds)
        logger.info(f"Tổng số dòng: {len(total_preds)}")
        logger.info(f"Số dòng bất thường: {num_anomaly}")
        logger.info(f"Tỉ lệ bất thường: {anomaly_rate:.2%}")
    else:
        logger.info("Không có dòng nào để tính tỉ lệ bất thường.")

    # --- Lưu kết quả vào file JSON ---
    output_json_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "inference_results.json"
    )
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(log_jsons, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu kết quả vào {output_json_path}")

      # In ra 5 log_json đầu tiên để kiểm tra
    # --- Gửi log_jsons lên Elasticsearch ---
    logger.info("Đang gửi log lên Elasticsearch...")
    # send_logs_to_elasticsearch(log_jsons)
    logger.info("Đã gửi xong log lên Elasticsearch.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_log', required=True, help='Path to new log file')
    parser.add_argument('--drain_json', required=True, help='Path to trained drain.json')
    parser.add_argument('--model_h5', required=True, help='Path to trained logkey_model.h5')
    parser.add_argument('--config_json', required=True, help='Path to deeplog_conf.json')
    parser.add_argument('--drain_conf', required=True, help='Path to drain.conf (ini) used for session splitting')

    parser.add_argument('--es_host', default="http://localhost:9200", help='Elasticsearch host')
    parser.add_argument('--es_index', default="logs", help='Elasticsearch index name')
    args = parser.parse_args()
    main(args)