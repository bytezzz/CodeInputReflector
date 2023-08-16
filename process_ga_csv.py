import pathlib
import pandas as pd
import json

transformed = []

for csv_file in pathlib.Path('ga_csv').glob('*.csv'):
    df = pd.read_csv(csv_file)
    df = df[df['Is Success'] == 1]
    for _, row in df.iterrows():
        try:
            transformed.append({'ori_func':row['Original Code'], 'ori_label':row['Original Prediction'], 'adv_label':row['Adv Prediction'], 'adv_func':row['Adversarial Code']})
        except:
            pass

print(len(transformed))
with open('adv_examples.jsonl','w') as f:
    for item in transformed:
        f.write(json.dumps(item) + '\n')

