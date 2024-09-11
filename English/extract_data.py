# 1. packages
import re
import os
import numpy as np
import pandas as pd

# 2. functions
def find_files(directory, extension):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list

    
# find all files
filenames = find_files('GCDC_Corpus_v2', '.csv')

# read and concat
df = pd.DataFrame()
for filename in filenames:
    filedata = pd.read_csv(filename)
    filedata['source'] = filename.split('\\')[-1].split('_')[0]
    if filename.split('\\')[-1].split('_')[0] == 'Yahoo':
        filedata['subject'] =  filedata['question_title']
    df = pd.concat([df, filedata])
    
    
df['subject'] = df['subject'].apply(lambda x: str(x).replace('\n', ' ') if x==x else np.nan)
df['subject'] = df['subject'].apply(lambda x: re.sub(' +', ' ', str(x)) if x==x else np.nan)

df['text'] = df['text'].apply(lambda x: x.replace('\n', ' '))
df['text'] = df['text'].apply(lambda x: re.sub(' +', ' ', x))
df.index = range(len(df))

df['coherence'] = df['labelA']
df['lang'] = 'en'
df['index'] = df['text_id']
df = df[['index', 'subject', 'text', 'coherence', 'source', 'lang']]

df.to_csv('summary.csv', index=False)
