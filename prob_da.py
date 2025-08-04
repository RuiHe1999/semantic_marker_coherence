# 1. packages
import numpy as np
import pandas as pd
from tqdm import tqdm 

import spacy
spacy.prefer_gpu()
nlp = spacy.load('da_core_news_lg')

import torch
from torch.nn import functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from transformers import BertTokenizer, BertForNextSentencePrediction, AutoTokenizer, AutoModelForCausalLM
# 2. consonants
bert_tokenizer = BertTokenizer.from_pretrained('Maltehb/danish-bert-botxo')
bert_model = BertForNextSentencePrediction.from_pretrained('Maltehb/danish-bert-botxo')
bert_model = bert_model.to(device)

huggingface_token = ''
model_name = "danish-foundation-models/munin-7b-alpha"
mistral_tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
mistral_model = AutoModelForCausalLM.from_pretrained(model_name, token=huggingface_token)
mistral_model = mistral_model.to(device)

# 3. functions
def sent_ppl(sents):
    
    # NSP surprisal
    nsp_probs = []
    for i in range(len(sents) -1):
        inputs = bert_tokenizer.encode_plus(sents[i], sents[i+1], 
                                            return_tensors='pt', 
                                            padding=True, 
                                            truncation=True).to(device)
        outputs = bert_model(**inputs)
        nsp_probs.append(F.softmax(outputs.logits, dim=-1).cpu().detach().numpy()[0][0])

    surprisals = [-np.log2(prob) for prob in nsp_probs]
   
    return np.exp2(np.mean(surprisals))

def word_ppl(sentence):

    # Encode the sentence using the tokenizer
    input_ids = mistral_tokenizer.encode(sentence, return_tensors='pt').to(device)
    loss = mistral_model(input_ids, labels=input_ids).loss
    mistral_ppl = np.exp2(loss.item())
    
    return mistral_ppl

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
    result['BERT_NSP_PPL'] = sent_ppl(sents)
    result['Word_PPL'] = word_ppl(' '.join(sents))
    
    results = pd.concat([results, result])
    
results.to_csv('features/da_prob.csv', index=False)
    


    
    

    































