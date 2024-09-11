# 1. packages
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm 

# 2. functions
def find_files(directory, extension):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list
    
# find all files
filenames = find_files('ddisco', '.tsv')

# read and concat
df = pd.DataFrame()
for filename in filenames:
    filedata = pd.read_csv(filename, sep='\t')
    filedata['coherence'] = filedata['rating']
    filedata['source'] = filedata['domain']
    df = pd.concat([df, filedata])
    
    
new_df = pd.DataFrame()
for _, row in tqdm(df.iterrows(), total=len(df)):
    source = row['source']
    if source == 'wikipedia':
        text = row['text']
        row['subject'] = text.split('  ')[0]
        row['text'] =  re.sub(' +', ' ', ' '.join(text.split('  ')[1:]).replace('\n', ' '))
        new_df = pd.concat([new_df, pd.DataFrame(row).T])
    else:
        text = row['text']
        row['subject'] = np.nan
        row['text'] =  text
        new_df = pd.concat([new_df, pd.DataFrame(row).T])

new_df['lang'] = 'da'
new_df['index'] = [f'ddisco_wiki_{x+1}' for x in range(len(new_df))]
new_df = new_df[['index', 'subject', 'text', 'coherence', 'source', 'lang']]

new_df.to_csv('summary.csv', index=False)
