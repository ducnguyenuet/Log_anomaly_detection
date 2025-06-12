import time
from kafka import KafkaConsumer
import json

KAFKA_TOPIC = 'filebeat-logs'
KAFKA_BROKERS = 'localhost:9092'

time.sleep(20)

RETRY_COUNT = 5
RETRY_DELAY = 10

for i in range(RETRY_COUNT):
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=[KAFKA_BROKERS],
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='my-host-log-consumer-group-root',
            value_deserializer=lambda x: x.decode('utf-8')
        )

        count = 0
        start_time = time.time()
        for message in consumer:
            count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                print(count)
                count = 0
                start_time = time.time()
        break

    except Exception as e:
        time.sleep(RETRY_DELAY)