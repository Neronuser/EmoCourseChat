import pandas as pd
import pandas_profiling

DATA_FILE = "data/processed/dialogue/full_dialogues_labeled.csv"
REPORT_FILE = "reports/conversation_emotion.html"

if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE, encoding='UTF-8', header=None, usecols=[2], names=['emotion'])
    processed_report = pandas_profiling.ProfileReport(df)
    processed_report.to_file(REPORT_FILE)
