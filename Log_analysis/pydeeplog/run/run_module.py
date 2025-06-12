import time
import json
import subprocess
from kafka import KafkaConsumer
import sys
import os

KAFKA_TOPIC = 'filebeat-logs'
KAFKA_BROKERS = 'localhost:9092'
BATCH_SIZE = 5000

def run_inference(log_file, idx):
    python_executable = sys.executable 
    cmd = [
        python_executable, "-m", "run.run_inference",
        "--input_log", log_file,
        "--drain_json", "artifacts/deeplog_model/drain.json",
        "--model_h5", "artifacts/deeplog_model/logkey_model.h5",
        "--config_json", "artifacts/deeplog_model/deeplog_conf.json",
        "--drain_conf", "hdfs_config.ini",
        "--es_index", "test_index",
        "--es_host", "localhost"
    ]
    print(f"Chạy inference cho {log_file} ...")
    subprocess.run(cmd)

def main():
    print(f"Kết nối tới Kafka: {KAFKA_BROKERS}, topic: {KAFKA_TOPIC}")
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_BROKERS],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='log-analysis-consumer',
        value_deserializer=lambda x: x.decode('utf-8')
    )

    batch = []
    file_idx = 1

    for message in consumer:
        try:
            log_data = json.loads(message.value)
            log_line = log_data['message'] if 'message' in log_data else str(log_data)
        except Exception:
            log_line = message.value

        print(f"Nhận được log: {log_line.strip()}")
        batch.append(log_line)
        if len(batch) >= BATCH_SIZE:
            log_filename = f"data/hdfs{file_idx}.log"
            with open(log_filename, "w", encoding="utf-8") as f:
                for line in batch:
                    f.write(line.strip() + "\n")
            print(f"Đã ghi {len(batch)} dòng vào {log_filename}")
            run_inference(log_filename, file_idx)
            batch = []
            file_idx += 1

if __name__ == "__main__":
    main()