# 1. packages
import numpy as np
import pandas as pd
from tqdm import tqdm 

import spacy
spacy.prefer_gpu()
nlp = spacy.load('zh_core_web_lg')

from transformers import AutoTokenizer, AutoModel
sbert_tokenizer = AutoTokenizer.from_pretrained('shibing624/text2vec-base-chinese')
sbert_model = AutoModel.from_pretrained('shibing624/text2vec-base-chinese')

from util import compute_wave_features, get_sbert_embedding, subject_sim

import warnings
warnings.filterwarnings("ignore")

# 2. consonants
wav_feats = ['MeanK1', 'MeanK2', 'Global', 'MCR', 'SSC', 'WL', 'Var', 'Peak', 
             'Valley', 'Amp', 'Skew', 'Kurt', 'ApEn', 'Acf', 'AcfZcr']

# 3. commands
data = pd.read_csv('Chinese/summary.csv')

results = pd.DataFrame()
for _, row in tqdm(data.iterrows(), total=len(data)):
    text_id, subject, text, coherence, source, lang = row
    text = ''.join(eval(text))
    
    # preprocess text 
    doc = nlp(text)
    sents = [sent.text for sent in doc.sents]
    
    result = pd.DataFrame([[text_id, coherence, source, lang]], 
                          columns=['Index', 'coherence', 'source', 'lang'])
    
    if subject != subject:
        result[[f'Prompt_{feat}' for feat in wav_feats]] = [np.nan] * len(wav_feats)
        
    elif len(sents) < 4:
        result[[f'Prompt_{feat}' for feat in wav_feats]] = [np.nan] * len(wav_feats)
        
    else:
        # embedding
        sent_embeds = get_sbert_embedding([subject]+sents, sbert_tokenizer, sbert_model, max_length=512)
        
        # compare to subject embedding 
        feats = compute_wave_features(subject_sim(sent_embeds))
        feats.insert(1, np.nan)
        
        prt2text = subject_sim(
            get_sbert_embedding([subject, text], sbert_tokenizer, sbert_model, max_length=512)
            )
        feats.insert(2, prt2text.item())
        
        result[[f'Prompt_{feat}' for feat in wav_feats]] = feats
    
    results = pd.concat([results, result])
        
        
results.to_csv('features/zh_prompt_sim.csv', index=False)        
    
    
    


    
    

    































