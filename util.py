import torch
import torch.nn.functional as F
from torch import Tensor

import numpy as np
import networkx as nx
import antropy as ant
from scipy import stats
from scipy.spatial import distance
from statsmodels.tsa.stattools import acf

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_sbert_embedding(sents, tokenizer, model, max_length=8192):
    
    # Tokenize the input texts
    batch_dict = tokenizer(sents, max_length=max_length, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = outputs.last_hidden_state[:, 0]
     
    # (Optionally) normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.detach().numpy()
    
def get_bert_embedding(sents, tokenizer, model):
    
    embeddings = []
    for sent in sents:
        
        encoded_input = tokenizer(sent, return_tensors='pt', truncation=True)
        output, pooler_output = model(input_ids=encoded_input['input_ids'],
                                      attention_mask=encoded_input['attention_mask'],
                                      return_dict=False)
        embeddings.extend(output[0].tolist()) 
    
    embeddings = torch.Tensor(embeddings)
    embeddings = F.normalize(embeddings, p=2, dim=1)
        
    return embeddings.numpy()

def slope_sign_changes(data):
    
    ssc = 0
    for i in list(range(len(data)-1))[1:]:
        if (data[i] < data[i+1]) and (data[i] < data[i-1]):
            ssc += 1
        elif (data[i] > data[i+1]) and (data[i] > data[i-1]):
            ssc += 1
        else:
            ssc += 0
    return ssc

def consecK(embeddings, k):
    
    assert (k >= 1)
    
    # k order sem sim
    consecs_k = []
    for i in range(len(embeddings)-k):
        embed1 = embeddings[i]
        embed2 = embeddings[i+k]
        consec_k = 1 - distance.cosine(embed1, embed2)
        consecs_k.append(consec_k)
    
    return np.array(consecs_k)

def global_similarity(embeddings):

    sem_sim = 1 - distance.cdist(embeddings, embeddings, metric='cosine')
    mean_g = sem_sim[np.tril_indices(sem_sim.shape[0], k=-1)]
        
    return np.nanmean(mean_g)
       
def centroid_stat(embeddings):
    
    centroid = np.mean(embeddings, axis=0)
    
    stat_cent_sims = []
    for i in range(len(embeddings)):
        embed = embeddings[i]
        stat_cent_sim = 1 - distance.cosine(embed, centroid)
        stat_cent_sims.append(stat_cent_sim)
    
    return np.array(stat_cent_sims)
    
def centroid_cuml(embeddings):

    cuml_cent_sims = []
    for i in range(1, len(embeddings)):
        embed = embeddings[i]
        centroid = np.mean(embeddings[:i, :], axis=0)
        cuml_cent_sim = 1 - distance.cosine(embed, centroid)
        cuml_cent_sims.append(cuml_cent_sim)
    
    return np.array(cuml_cent_sims)

def subject_sim(embeddings):
       
    # k order sem sim
    sims = []
    for i in range(1, len(embeddings)):
        embed1 = embeddings[0]
        embed2 = embeddings[i]
        sim = 1 - distance.cosine(embed1, embed2)
        sims.append(sim)
    
    return np.array(sims)

def compute_wave_features(sim_wav):
        
    assert(np.all(sim_wav == sim_wav))
       
    # mean
    mean = np.mean(sim_wav)
    
    # variance
    var = np.var(sim_wav)
    # peak
    peak = np.max(sim_wav)
    # valley
    valley = np.min(sim_wav)
    # amplitude
    amplitude = peak - valley
    # skewness
    skew = stats.skew(sim_wav)
    # excess kurtosis
    kurt = stats.kurtosis(sim_wav) - 3
    # mean crossing rate
    mcr = ((np.diff(np.sign(sim_wav-np.mean(sim_wav))) != 0).sum() - ((sim_wav-np.mean(sim_wav)) == 0).sum()) / (len(sim_wav) - 1)
    # normalized slope sign changes
    ssc = slope_sign_changes(sim_wav) / (len(sim_wav) - 2)
    # waveform length
    wl = np.mean([np.abs(sim_wav[i+1] - sim_wav[i]) for i in range(len(sim_wav)-1)])
    # entropy
    apen = ant.app_entropy(sim_wav)
    # ACW
    acw = acf(sim_wav, nlags=len(sim_wav)-1, qstat=False, alpha=None, fft=False)
    acw_zcr = ((np.diff(np.sign(acw)) != 0).sum() - (acw == 0).sum()) / (len(acw) - 1)
    
    feature = [mean, mcr, ssc, wl, var, peak, valley, amplitude, skew, kurt, apen, acw[1], acw_zcr]
     
    return feature

def threshold_proportional(W, p, copy=True):
    """
    brainconn downloaded from https://github.com/fiuneuro/brainconn
    
    Modified from the "matrix.py" script from brainconn:
    1. We set all negative values to zeros too before thresholding
    2. We threshold the top p percentage of coefficients per row, not in the 
    whole matrix
    3. We used np.ceil instead of np.round for thresholding numbers to avoid
    singular matrix which is harmful for later analysis
    4. If there are more than one smallest values in thresholding (e.g. in a 
    row where the smallest value to enter threshold is 0.3 but there are three
    0.3 in the row), all smallest values enter the threshold. In the original 
    funtion, only the last one entered. 
    
    This function "thresholds" the connectivity matrix by preserving a
    proportion p (0<p<1) of the strongest weights. All other weights are \
    set to 0.

    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : :obj:`numpy.ndarray`
        weighted connectivity matrix
    p : float
        proportional weight threshold (0<p<1)
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : :obj:`numpy.ndarray`
        thresholded connectivity matrix

   
    """
    if p > 1 or p < 0:
        raise ValueError('Threshold must be value between 0 and 1')
    if copy:
        W = W.copy()
        
    # number of nodes
    n = len(W)		
    # clear diagonal				
    np.fill_diagonal(W, 0)			

    # if symmetric matrix
    # ensure symmetry is preserved
    if np.allclose(W, W.T):				
        W[np.tril_indices(n)] = 0
        # halve number of removed links	
        ud = 2						
    else:
        ud = 1
        
    W[W<0]=0    	

    # thresholding  
    W_thresholded = np.zeros(W.shape)
    for i in range(W.shape[0]):
        W_unit = W[i, :]
        #  find all links
        ind = np.where(W_unit)[0]
        # sort indices by magnitude
        sorted_values = np.argsort(W_unit[ind])[::-1]
        # number of links to be preserved
        en = int(np.ceil(((n - i - 1) * p) / ud))    
        # values to maintain
        ind_thre = ind[sorted_values][:en+1]
        # find if there are any same value 
        if ind_thre.shape[0] != 0:
            for t in ind[sorted_values][en:]:
                if W_unit[ind_thre[-1]] == W_unit[t]:
                    ind_thre = np.append(ind_thre, t)
      
        # apply threshold
        W_unit[ind[np.isin(ind, ind_thre, invert=True)]] = 0
        W_thresholded[i, :] = W_unit
     
    # # if symmetric matrix
    if ud == 2:
        W[:, :] = W_thresholded + W_thresholded.T	
    else:
        W[:, :] = W

    return W

def select_threshold(affinity_matrix, t=0.55):
    
    '''
    Threshold reference:
    Watts, D., Strogatz, S. Collective dynamics of ‘small-world’ networks. 
    Nature 393, 440–442 (1998). https://doi.org/10.1038/30918
    
    
    We DO NOT apply the 1.1 as minimum sigma as this is a value derived from
    https://www.sciencedirect.com/science/article/pii/S0006322311005476#sec5,
    i.e. from their spefcific data, to control the threshold lower than 0.34, 
    which means it's NOT necessarily applicable to all data, e.g. the London 
    speech data. We choose a range of threshold from 0.05 to 0.55 and pick up 
    the lowest to control that. 
    
    '''
    
    N = affinity_matrix.shape[0]
    min_avg_degree = 2 * np.log(N)
    threshold = 0.05

    while threshold <= t:
        # threshold matrix
        affinity_threshold = threshold_proportional(affinity_matrix, threshold, copy=True)
        
        # convert to undirected graph
        G = nx.Graph()

        nodes = range(len(affinity_threshold))
        G.add_nodes_from(nodes)

        for i in range(affinity_threshold.shape[0]):
            for j in range(affinity_threshold.shape[1]):
                weight = affinity_threshold[i][j]  
                if weight != 0:
                    # G.add_edge(i, j, weight=weight)  
                    G.add_edge(i, j)
                    
        avg_degree = sum(dict(G.degree()).values()) / len(G)
        
        # print(threshold, avg_degree, min_avg_degree)

        if avg_degree > min_avg_degree:
            return threshold, G

        threshold += 0.05
        
    return None

def extract_graph_features(embeddings):
    
    affinity = 1 - distance.cdist(embeddings, embeddings, metric='cosine')
    np.fill_diagonal(affinity, 0)
    
    # threshold the affinity matrix
    t, G = select_threshold(affinity, t=0.8)
        
    # centrality 
    cc = np.mean(list(nx.closeness_centrality(G).values()))
    
    # clustering coefficient
    clustering = nx.average_clustering(G)
    
    return [cc, clustering]

    # small_worldness
    # small_worldness = nx.sigma(G, niter=5, nrand=10, seed=7)
    
    # return [cc, clustering, small_worldness]
    









