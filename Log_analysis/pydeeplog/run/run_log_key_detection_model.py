import argparse
import os

from pathlib import Path
from tensorflow.keras.models import load_model
from deeplog_trainer.model.model_evaluator import ModelEvaluator
from deeplog_trainer.model.model_manager import ModelManager
from deeplog_trainer.model.training import ModelTrainer

from . import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')


def run_log_key_model(logger, output_path, window_size,
                      lstm_units, max_epochs, train_dataset, val_dataset,
                      data_preprocess: DataPreprocess, early_stop, batch_size,
                      out_tensorboard_path, top_k):
    num_tokens = data_preprocess.get_num_tokens()
    model_manager = ModelManager()
    model = model_manager.build(ModelManager.MODEL_TYPE_LOG_KEYS,
                                input_size=window_size,
                                lstm_units=lstm_units,
                                num_tokens=num_tokens)
    model.summary()
    # print(data_preprocess)
    X_train, y_train = data_preprocess.transform(
        data_preprocess.chunks(train_dataset, window_size=window_size),
        add_padding=window_size
    )
    print(X_train)
    X_val, y_val = data_preprocess.transform(
        data_preprocess.chunks(val_dataset, window_size=window_size),
        add_padding=window_size
    )
    # model_path = r'D:\VDT_SE_PROJECT\Log_analysis\pydeeplog\artifacts\deeplog_model\logkey_model.h5'

    # print(os.path.exists(model_path))  # In ra True nếu file tồn tại
    # model = load_model(model_path)
    model_trainer = ModelTrainer(logger, epochs=max_epochs,
                                 early_stop=early_stop,
                                 batch_size=batch_size)
    # Run training and validation to fit the model
    model_trainer.train(model, [X_train, y_train], [X_val, y_val],
                        out_tensorboard_path=out_tensorboard_path)
    # Save the model
    model_manager.save(model, output_path, 'logkey_model.h5')
    # Calculate scores for different K values in the validation set
    model_evaluator = ModelEvaluator(model, top_k=top_k)
    scores = model_evaluator.compute_scores(X_val, y_val)
    logger.info('-' * 10 + ' K = ' + str(top_k) + ' ' + '-' * 10)
    logger.info('- Num. items: {}'.format(scores['n_items']))
    logger.info('- Num. normal: {}'.format(scores['n_correct']))
    logger.info('- Accuracy: {:.4f}'.format(scores['accuracy']))
    y_pred = model_evaluator.predict(X_val)
    for idx, (true_label, pred_label) in enumerate(zip(y_val, y_pred)):
        # Lấy index của nhãn đúng từ vector one-hot
        true_idx = int(np.argmax(true_label))
        # Nếu nhãn đúng nằm trong top-k dự đoán
        if true_idx in pred_label:
            status = "Bình thường"
        else:
            status = "Bất thường"
        logger.info(f"Mẫu {idx}: Dự đoán = {pred_label}, Thực tế = {true_idx} => {status}")
    # Save config values in a json file:
    with open(os.path.join(output_path, 'deeplog_conf.json'), 'w') as f:
        par = dict(num_templates=num_tokens - 3, top_candidates=top_k,
                   window_size=window_size)
        json.dump(par, f)
    # Save empty workflow in json file
    with open(os.path.join(output_path, 'workflows.json'), 'w') as f:
        network_dict = {"root": {"value": None,
                                 "children": {},
                                 "parents": [],
                                 "is_start": False,
                                 "is_end": False}}
        json.dump(network_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_log_key_model_runner_args(parser)

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    try:
        os.makedirs(args.output_path, exist_ok=True)
    except OSError as error:
        logger.error("Directory {} can not be created".format(args.output_path))
        exit(1)
    train_dataset, val_dataset, test_dataset, data_preprocess = create_datasets(
        logger, args.input_path, args.window_size, args.train_ratio,
        args.val_ratio)
    run_log_key_model(logger, args.output_path, args.window_size,
                      args.lstm_units, args.max_epochs, train_dataset,
                      val_dataset, data_preprocess, args.early_stop,
                      args.batch_size, args.out_tensorboard_path, args.top_k)
