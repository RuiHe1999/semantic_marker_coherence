# 1. packages
import numpy as np
import pandas as pd
from tqdm import tqdm 

import fasttext
ft = fasttext.load_model('cc.da.300.bin')

# from gensim.models import fasttext
# ft = fasttext.load_facebook_vectors('E:/A-Horace/PhD/FastText/cc.da.300.bin')

import spacy
spacy.prefer_gpu()
nlp = spacy.load('da_core_news_lg')

from transformers import BertTokenizer, BertModel
bert_tokenizer = BertTokenizer.from_pretrained('Maltehb/danish-bert-botxo')
bert_model = BertModel.from_pretrained("Maltehb/danish-bert-botxo")

from transformers import AutoTokenizer, AutoModel
sbert_tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-multilingual-base')
sbert_model = AutoModel.from_pretrained('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)


import torch
import torch.nn.functional as F

from util import (get_sbert_embedding, get_bert_embedding,
                  consecK, compute_wave_features, 
                  centroid_stat, centroid_cuml, global_similarity)

import warnings
warnings.filterwarnings("ignore")

# 2. consonants
wav_feats = ['MeanK1', 'MeanK2', 'Global', 'MCR', 'SSC', 'WL', 'Var', 'Peak', 
             'Valley', 'Amp', 'Skew', 'Kurt', 'ApEn', 'Acf', 'AcfZcr']

# 3. functions
def extract_wav_feats(embeds): 
    
    if len(embeds) <= 3:
        data = [np.nan] * (len(wav_feats) * 3)
    else:
        # consecutive semantic similarity 
        consec_feats = compute_wave_features(consecK(embeds, 1))
        consec_feats.insert(1, np.mean(consecK(embeds, 2)))
        consec_feats.insert(2, global_similarity(embeds))
        
        # similarity between each word and the static centroid
        cent_stat = compute_wave_features(centroid_stat(embeds))
        cent_stat.insert(1, np.nan)
        cent_stat.insert(2, np.nan)
        
        # similarity between each word and the cumulative centroids
        cent_cuml = compute_wave_features(centroid_cuml(embeds))
        cent_cuml.insert(1, np.nan)
        cent_cuml.insert(2, np.nan)
        
        data = consec_feats + cent_stat + cent_cuml
    
    return data
      
# 4. commands
data = pd.read_csv('Danish/summary.csv')

results = pd.DataFrame()
for _, row in tqdm(data.iterrows(), total=len(data)):
    text_id, subject, text, coherence, source, lang = row
    
    result = pd.DataFrame([[text_id, coherence, source, lang]], 
                          columns=['Index', 'coherence', 'source', 'lang'])
    
    # preprocess text 
    doc = nlp(text)
    sents = [sent.text for sent in doc.sents]
    tokens = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
    tags = [token.pos_ for token in doc]

    # embedding
    # ft_embeds = np.array([ft[token] for token in tokens])
    ft_embeds = np.array([ft.get_word_vector(token) for token in tokens])
    ft_embeds = F.normalize(torch.tensor(ft_embeds), p=2, dim=1).numpy()
    bert_embeds = get_bert_embedding(sents, bert_tokenizer, bert_model)
    sent_embeds = get_sbert_embedding(sents, sbert_tokenizer, sbert_model, max_length=8192)
    
    # compute sem sim between consecutive pairs, with stactic/cumulative centroids
    result[[f'FT{tag}_{feat}' for tag in ['', '_stat', '_cuml'] for feat in wav_feats]] = extract_wav_feats(ft_embeds)
    result[[f'BERT{tag}_{feat}' for tag in ['', '_stat', '_cuml'] for feat in wav_feats]] = extract_wav_feats(bert_embeds)
    result[[f'Sent{tag}_{feat}' for tag in ['', '_stat', '_cuml'] for feat in wav_feats]] = extract_wav_feats(sent_embeds)
    
    results = pd.concat([results, result])
    
results.to_csv('features/da_wav_sim.csv', index=False)
    
    
    


    
    

    































