import csv
import subprocess

if __name__ == '__main__':
    INFERENCE_FILE = 'data/processed/dialogue/fasttext.csv'
    MODEL_PATH = 'models/emotion_classification/fasttext/model.bin'
    OUTPUT_PATH = 'data/processed/dialogue/fasttext_classified.csv'
    TEMP_FILE = 'data/processed/dialogue/fasttext_temp.csv'
    label_prefix = '__label__'

    preds = subprocess.check_output(['./fastText-0.1.0/fasttext', 'predict', MODEL_PATH, INFERENCE_FILE])
    label_len = len(label_prefix)
    text_preds = (pred[label_len:] for pred in preds.decode("utf-8").split('\n'))
    with open(OUTPUT_PATH, 'w') as output_file:
        writer = csv.writer(output_file)
        for pred in text_preds:
            writer.writerow([pred])
