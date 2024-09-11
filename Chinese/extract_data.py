# 1. packages
import json
import pandas as pd



def custom_sort(item):
    return (item['paragraphIdx'], item['sentenceIdx'])

with open('train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

with open('test.json', 'r', encoding='utf-8') as file:
    data.extend(json.load(file))
    
df = pd.DataFrame(data)
df['subject'] = df['title']
df['coherence'] = df['logicGrade'] + 1
df['text'] = df['text'].apply(lambda x: str(x))
df['source'] = 'CEDCC'
df['lang'] = 'zh'
df['index'] = df['id']
df = df[['index', 'subject', 'text', 'coherence', 'source', 'lang']]
df.to_csv('summary.csv', index=False, encoding='utf-8-sig')

