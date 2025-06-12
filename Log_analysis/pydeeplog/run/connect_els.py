from elasticsearch import Elasticsearch, helpers
import json
from datetime import datetime

CLOUD_ID = "Logs:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyQ3NjZmMjFkNzAyYWM0ZmFiYjQ4N2E3MzdkZWZkNzRjZiQyNDUyZDlkODNmZjQ0YmVlOWIyMzE5OTFhOTkxMTFjZA=="
CLOUD_AUTH = ("elastic","oRARoz6PxOdHDlOXBe0MZ0uu")

def read_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # Đây là một list các object
        return data
    
def convert_timestamp_to_iso(ts):
    try:
        # Format gốc là "yyMMdd HHmmss"
        return datetime.strptime(ts, "%y%m%d %H%M%S").isoformat()
    except Exception as e:
        print(f"Lỗi chuyển timestamp: {ts} - {e}")
        return None

def prepare_logs(log_jsons):
    for log in log_jsons:
        if "timestamp" in log:
            iso_timestamp = convert_timestamp_to_iso(log["timestamp"])
            if iso_timestamp:
                log["@timestamp"] = iso_timestamp  # Elasticsearch nhận diện được
    return log_jsons

def send_logs_to_elasticsearch(log_jsons, cloud_id=CLOUD_ID, cloud_auth=CLOUD_AUTH, index_name="logs"):
    # print(log_jsons[:500])
    es = Elasticsearch(cloud_id=cloud_id, basic_auth=cloud_auth)
    prepared_logs = prepare_logs(log_jsons)

    actions = [
        {
            "_index": index_name,
            "_source": log_json
        }
        for log_json in prepared_logs if isinstance(log_json, dict) 
    ]
    helpers.bulk(es, actions)

if __name__ == "__main__":
    # Example usage
    log_jsons = read_json_file('../data/inference_results.json')
    send_logs_to_elasticsearch(log_jsons)
    print("Logs sent to Elasticsearch successfully.")