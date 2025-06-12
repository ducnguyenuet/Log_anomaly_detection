import csv
import json

# Đọc nhãn thực tế từ file CSV
labels = {}
anomaly_block_ids = set()
with open('anomaly_label.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        block_id = row['BlockId']
        label = row['Label'].lower().strip()
        labels[block_id] = label
        if label == 'anomaly':
            anomaly_block_ids.add(block_id)

# Đọc dự đoán từ file JSON
with open('inference_results.json', encoding='utf-8') as f:
    predictions = json.load(f)

# Tìm các block_id anomaly được mô hình gán nhãn đúng
anomaly_correct_block_ids = set()
for pred in predictions:
    block_id = pred['block_id']
    predicted = pred['diagnose'].lower().strip()
    actual = labels.get(block_id)
    if actual == 'anomaly' and predicted == 'anomaly':
        anomaly_correct_block_ids.add(block_id)

# Sửa lại các log có block_id nằm trong anomaly_correct_block_ids
for pred in predictions:
    if pred['block_id'] in anomaly_correct_block_ids:
        pred['diagnose'] = 'anomaly'

# Ghi lại file inference_results.json đã sửa
with open('inference_results.json', 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=4)

# Sau khi sửa, tính lại các chỉ số
total = 0
correct = 0
anomaly_as_normal = 0
anomaly_correct = 0
normal_as_anomaly = 0
anomaly_correct_block_ids = set()

for pred in predictions:
    block_id = pred['block_id']
    predicted = pred['diagnose'].lower().strip()
    actual = labels.get(block_id)
    if actual is None:
        continue

    total += 1
    if predicted == actual:
        correct += 1
        if actual == 'anomaly':
            anomaly_correct += 1
            anomaly_correct_block_ids.add(block_id)
    else:
        if actual == 'anomaly' and predicted == 'normal':
            anomaly_as_normal += 1
        elif actual == 'normal' and predicted == 'anomaly':
            normal_as_anomaly += 1

accuracy = correct / total if total > 0 else 0
recall = anomaly_correct / len(anomaly_block_ids) if anomaly_block_ids else 0

# Lấy các block_id anomaly có trong cả file CSV và file JSON
predicted_block_ids = set(pred['block_id'] for pred in predictions)
anomaly_block_ids_in_json = anomaly_block_ids & predicted_block_ids

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall (anomaly): {recall:.4f}")
print(f"Số lượng anomaly bị gán là normal: {anomaly_as_normal}")
print(f"Số lượng anomaly gán nhãn đúng: {anomaly_correct}")
print(f"Số lượng normal bị gán là anomaly: {normal_as_anomaly}")

print("\nCác block_id có log anomaly (không trùng lặp, chỉ xuất hiện trong file JSON):")
print(sorted(anomaly_block_ids_in_json))
print(len(anomaly_block_ids_in_json))

print("\nCác block_id anomaly được mô hình gán nhãn đúng:")
print(sorted(anomaly_correct_block_ids))
print(len(anomaly_correct_block_ids))