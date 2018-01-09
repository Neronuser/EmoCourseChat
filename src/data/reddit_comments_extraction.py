import csv
import json

DATA_PATH = 'data/raw/dialogue/reddit_comments_month'
OUTPUT_PATH = 'data/processed/dialogue/reddit.csv'

if __name__ == '__main__':
    """ https://www.kaggle.com/data/31657 """
    id2info = {}
    with open(DATA_PATH) as data_file:
        for i, line in enumerate(data_file):
            comment = json.loads(line)
            body = comment['body']
            if len(body.split()) < 40:
                id2info[comment["name"]] = (comment["parent_id"], body.replace('\n', ''))
    print("Loaded")
    with open(OUTPUT_PATH, 'w') as out_file:
        writer = csv.writer(out_file, quoting=csv.QUOTE_MINIMAL)
        for i, (key, value) in enumerate(id2info.items()):
            parent = id2info.get(value[0])
            if parent:
                writer.writerow([parent[1], value[1]])
