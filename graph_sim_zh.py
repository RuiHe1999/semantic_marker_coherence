# 1. packages
import numpy as np
import pandas as pd
from tqdm import tqdm 

import fasttext
ft = fasttext.load_model('cc.zh.300.bin')

# from gensim.models import fasttext
# ft = fasttext.load_facebook_vectors('E:/A-Horace/PhD/FastText/cc.zh.300.bin')

import spacy
spacy.prefer_gpu()
nlp = spacy.load('zh_core_web_lg')

from transformers import BertTokenizer, BertModel
bert_tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')
bert_model = BertModel.from_pretrained("hfl/chinese-macbert-base")

from transformers import AutoTokenizer, AutoModel
sbert_tokenizer = AutoTokenizer.from_pretrained('shibing624/text2vec-base-chinese')
sbert_model = AutoModel.from_pretrained('shibing624/text2vec-base-chinese')

import torch
import torch.nn.functional as F

from util import get_sbert_embedding, get_bert_embedding, extract_graph_features

import warnings
warnings.filterwarnings("ignore")

# 2. consonants
graph_feats = ['CC', 'Cluster']

     
# 3. commands
data = pd.read_csv('Chinese/summary.csv')

results = pd.DataFrame()
for _, row in tqdm(data.iterrows(), total=len(data)):
    text_id, subject, text, coherence, source, lang = row
    text = ''.join(eval(text))
    
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
    sent_embeds = get_sbert_embedding(sents, sbert_tokenizer, sbert_model, max_length=512)
    
    result[[f'FT_{feat}' for feat in graph_feats]] = extract_graph_features(ft_embeds) if len(ft_embeds) >= 4 else [np.nan] * len(graph_feats) 
    result[[f'BERT_{feat}' for feat in graph_feats]] = extract_graph_features(bert_embeds) if len(bert_embeds) >= 4 else [np.nan] * len(graph_feats) 
    result[[f'Sent_{feat}' for feat in graph_feats]] = extract_graph_features(sent_embeds) if len(sent_embeds) >= 4 else [np.nan] * len(graph_feats) 
    results = pd.concat([results, result])
    
results.to_csv('features/zh_graph_sim.csv', index=False)
    
    
    


    
    

    































