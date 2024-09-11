# 1. packages
import numpy as np
import pandas as pd

import seaborn as sns
sns.set(font='Arial')

from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Arial"

import pingouin as pg
from statsmodels.stats.multitest import multipletests

# 2. consonants
dstr_feats = ['MeanK1', 'MeanK2', 'Global', 'Var', 'Peak', 'Valley', 'Amp', 'Skew', 'Kurt',]
wave_feats = ['MCR', 'SSC', 'WL', 'ApEn', 'Acf', 'AcfZcr']
grph_feats = ['CC', 'Cluster']
prob_feats = ['BERT_NSP_PPL', 'Word_PPL']

vmax = 0.3

# 3. functions
def evaluate_trend(coefs, sigfs):
    num_less_than_0_05 = np.sum(sigfs < 0.05)
    num_less_than_0_1 = np.sum(sigfs < 0.1)
    
    # pass at 0.05 level
    if (num_less_than_0_05 == 3) or (num_less_than_0_05 == 2 and num_less_than_0_1 == 3):
        significant_coefs = np.array(coefs)[sigfs < 0.05]
        if np.all(np.sign(significant_coefs) == np.sign(significant_coefs[0])):
            return "Pass"
        else:
            return "Fail"   
    
    # uncertain at 0.1 level 
    if num_less_than_0_1 >= 2:
        significant_coefs = np.array(coefs)[sigfs < 0.1]
        if np.all(np.sign(significant_coefs) == np.sign(significant_coefs[0])):
            return "Unc"
        else:
            return "Fail"
    
    # other as fail 
    return "Fail"  

def organize_corr_res(compare, feats, method='spearman', padjust=None, split=True):
    
    # read data    
    en_data = pd.read_csv(f'features/en_{compare}.csv')
    en_data = pd.get_dummies(en_data, columns=['source'])
    
    da_data = pd.read_csv(f'features/da_{compare}.csv')
    da_data = pd.get_dummies(da_data, columns=['source'])
    
    zh_data = pd.read_csv(f'features/zh_{compare}.csv')
    
    # correlation analyses
    en_corr = pg.pairwise_corr(en_data, columns=[['coherence'], feats], method=method, padjust=padjust)
    zh_corr = pg.pairwise_corr(zh_data, columns=[['coherence'], feats], method=method, padjust=padjust)
    da_corr = pg.pairwise_corr(da_data, columns=[['coherence'], feats], method=method, padjust=padjust)
    
    # add nan for feats without data
    for feat in feats:
        if feat not in en_corr.Y.values:
            add_corr = ['coherence', feat, method, 'two-sided', 0, np.nan, '', np.nan, np.nan]
            en_corr = pd.concat([en_corr, pd.DataFrame([add_corr], index=[0], columns=en_corr.columns)])
        
        if feat not in zh_corr.Y.values:
            add_corr = ['coherence', feat, method, 'two-sided', 0, np.nan, '', np.nan, np.nan]
            zh_corr = pd.concat([zh_corr, pd.DataFrame([add_corr], index=[0], columns=zh_corr.columns)])
        
        if feat not in da_corr.Y.values:
            add_corr = ['coherence', feat, method, 'two-sided', 0, np.nan, '', np.nan, np.nan]
            da_corr = pd.concat([da_corr, pd.DataFrame([add_corr], index=[0], columns=da_corr.columns)])
    
    # aggregate into one dataframe 
    en_corr.insert(0, 'Language', 'EN')
    da_corr.insert(0, 'Language', 'DA')
    zh_corr.insert(0, 'Language', 'ZH')
    results = pd.concat([en_corr, da_corr, zh_corr])
    results = results.reset_index(drop=True)
    
    # FDR by language
    results['p-corr'] = None
    for feat in feats:
        subset = results[results['Y'] == feat]
        _, p_corr, _, _ = multipletests(subset['p-unc'], method='fdr_bh')
        results.loc[subset.index, 'p-corr'] = p_corr

    # add singificance annotation
    annot_sig = lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '#' if p < 0.1 else ''
    results['significance'] = results['p-corr'].apply(annot_sig)
    results['r_filtered'] = np.where(results['p-corr'] < 0.1, results['r'], 0)
    results['formatted'] = results['r'].apply(lambda x: f'{x:.3f}' if x==x else '') + results['significance']
    
    # pivot table for boxplot
    rs = results.pivot(index='Language', columns='Y', values='r_filtered')[feats].reindex(['EN', 'ZH', 'DA'])
    rps = results.pivot(index='Language', columns='Y', values='formatted')[feats].reindex(['EN', 'ZH', 'DA'])
    
    # assess cross-lingual consistency on the trend  
    coefs = results.pivot(index='Language', columns='Y', values='r')[feats].reindex(['EN', 'ZH', 'DA'])
    sigfs = results.pivot(index='Language', columns='Y', values='p-corr')[feats].reindex(['EN', 'ZH', 'DA'])
    consistency = [evaluate_trend(coefs[feat], sigfs[feat]) for feat in feats]
    
    map_c = lambda c: 0 if c =='Fail' else vmax*0.4 if c =='Unc' else vmax*0.8 if c=='Pass' else ''
    consistency_  = [map_c(c) for c in consistency]
    
    rs.loc['Trend'] = consistency_
    rps.loc['Trend'] = consistency

    if split:
        rs.columns = [feat.split("_")[-1] for feat in feats]
        rps.columns = [feat.split("_")[-1] for feat in feats]
        
    return rs, rps

# 4. lexical category 
rs_ft, rps_ft = organize_corr_res('wav_sim', [f'FT_{feat}' for feat in dstr_feats + wave_feats])
rs_ft_stat, rps_ft_stat = organize_corr_res('wav_sim', [f'FT_stat_{feat}' for feat in dstr_feats + wave_feats])
rs_ft_cuml, rps_ft_cuml = organize_corr_res('wav_sim', [f'FT_cuml_{feat}' for feat in dstr_feats + wave_feats])

# visualize 
fig, axes = plt.subplots(3, 1, figsize=(18, 14))
axes = axes.flatten()

sns.heatmap(rs_ft, fmt="", annot=rps_ft,  annot_kws={"size": 12},
            cbar=True, cbar_kws={"shrink": 0.8, "orientation": 'vertical'}, 
            vmin =-vmax, vmax = vmax, 
            square=True, linewidth=.5, 
            cmap='coolwarm',
            ax=axes[0]
            )
axes[0].set(xlabel="", ylabel="")
axes[0].set_title('(A) Consecutive', loc='left')
axes[0].xaxis.set_ticks_position('top')  
axes[0].yaxis.set_tick_params(rotation=0)

sns.heatmap(rs_ft_stat, fmt="", annot=rps_ft_stat,  annot_kws={"size": 12},
            cbar=True, cbar_kws={"shrink": 0.8, "orientation": 'vertical'}, 
            vmin =-vmax, vmax = vmax, 
            square=True, linewidth=.5, 
            cmap='coolwarm',
            ax=axes[1]
            )
axes[1].set(xlabel="", ylabel="")
axes[1].set_title('(B) Static centroid', loc='left')
axes[1].xaxis.set_ticks_position('top')  
axes[1].yaxis.set_tick_params(rotation=0)

sns.heatmap(rs_ft_cuml, fmt="", annot=rps_ft_cuml,  annot_kws={"size": 12},
            cbar=True, cbar_kws={"shrink": 0.8, "orientation": 'vertical'}, 
            vmin =-vmax, vmax = vmax, 
            square=True, linewidth=.5, 
            cmap='coolwarm',
            ax=axes[2]
            )
axes[2].set(xlabel="", ylabel="")
axes[2].set_title('(C) Cumulative centroid', loc='left')
axes[2].xaxis.set_ticks_position('top')  
axes[2].yaxis.set_tick_params(rotation=0)

fig.savefig('results/FT_lexical_category.png', dpi=300, bbox_inches='tight') 

# 5. BERT tokens
rs_bert, rps_bert = organize_corr_res('wav_sim', [f'BERT_{feat}' for feat in dstr_feats + wave_feats])
rs_bert_stat, rps_bert_stat = organize_corr_res('wav_sim', [f'BERT_stat_{feat}' for feat in dstr_feats + wave_feats])
rs_bert_cuml, rps_bert_cuml = organize_corr_res('wav_sim', [f'BERT_cuml_{feat}' for feat in dstr_feats + wave_feats])

# visualize 
fig, axes = plt.subplots(3, 1, figsize=(18, 14))
axes = axes.flatten()

sns.heatmap(rs_bert, fmt="", annot=rps_bert,  annot_kws={"size": 12},
            cbar=True, cbar_kws={"shrink": 0.8, "orientation": 'vertical'}, 
            vmin =-vmax, vmax = vmax, 
            square=True, linewidth=.5, 
            cmap='coolwarm',
            ax=axes[0]
            )
axes[0].set(xlabel="", ylabel="")
axes[0].set_title('(A) Consecutive', loc='left')
axes[0].xaxis.set_ticks_position('top')  
axes[0].yaxis.set_tick_params(rotation=0)

sns.heatmap(rs_bert_stat, fmt="", annot=rps_bert_stat,  annot_kws={"size": 12},
            cbar=True, cbar_kws={"shrink": 0.8, "orientation": 'vertical'}, 
            vmin =-vmax, vmax = vmax, 
            square=True, linewidth=.5, 
            cmap='coolwarm',
            ax=axes[1]
            )
axes[1].set(xlabel="", ylabel="")
axes[1].set_title('(B) Static centroid', loc='left')
axes[1].xaxis.set_ticks_position('top')  
axes[1].yaxis.set_tick_params(rotation=0)

sns.heatmap(rs_bert_cuml, fmt="", annot=rps_bert_cuml,  annot_kws={"size": 12},
            cbar=True, cbar_kws={"shrink": 0.8, "orientation": 'vertical'}, 
            vmin =-vmax, vmax = vmax, 
            square=True, linewidth=.5, 
            cmap='coolwarm',
            ax=axes[2]
            )
axes[2].set(xlabel="", ylabel="")
axes[2].set_title('(C) Cumulative centroid', loc='left')
axes[2].xaxis.set_ticks_position('top')  
axes[2].yaxis.set_tick_params(rotation=0)

fig.savefig('results/BERT_tokens.png', dpi=300, bbox_inches='tight') 


# 6. Sentences
rs_sent, rps_sent = organize_corr_res('wav_sim', [f'Sent_{feat}' for feat in dstr_feats + wave_feats])
rs_sent_stat, rps_sent_stat = organize_corr_res('wav_sim', [f'Sent_stat_{feat}' for feat in dstr_feats + wave_feats])
rs_sent_cuml, rps_sent_cuml = organize_corr_res('wav_sim', [f'Sent_cuml_{feat}' for feat in dstr_feats + wave_feats])

# visualize 
fig, axes = plt.subplots(3, 1, figsize=(18, 14))
axes = axes.flatten()

sns.heatmap(rs_sent, fmt="", annot=rps_sent,  annot_kws={"size": 12},
            cbar=True, cbar_kws={"shrink": 0.8, "orientation": 'vertical'}, 
            vmin =-vmax, vmax = vmax, 
            square=True, linewidth=.5, 
            cmap='coolwarm',
            ax=axes[0]
            )
axes[0].set(xlabel="", ylabel="")
axes[0].set_title('(A) Consecutive', loc='left')
axes[0].xaxis.set_ticks_position('top')  
axes[0].yaxis.set_tick_params(rotation=0)

sns.heatmap(rs_sent_stat, fmt="", annot=rps_sent_stat,  annot_kws={"size": 12},
            cbar=True, cbar_kws={"shrink": 0.8, "orientation": 'vertical'}, 
            vmin =-vmax, vmax = vmax, 
            square=True, linewidth=.5, 
            cmap='coolwarm',
            ax=axes[1]
            )
axes[1].set(xlabel="", ylabel="")
axes[1].set_title('(B) Static centroid', loc='left')
axes[1].xaxis.set_ticks_position('top')  
axes[1].yaxis.set_tick_params(rotation=0)

sns.heatmap(rs_sent_cuml, fmt="", annot=rps_sent_cuml,  annot_kws={"size": 12},
            cbar=True, cbar_kws={"shrink": 0.8, "orientation": 'vertical'}, 
            vmin =-vmax, vmax = vmax, 
            square=True, linewidth=.5, 
            cmap='coolwarm',
            ax=axes[2]
            )
axes[2].set(xlabel="", ylabel="")
axes[2].set_title('(C) Cumulative centroid', loc='left')
axes[2].xaxis.set_ticks_position('top')  
axes[2].yaxis.set_tick_params(rotation=0)

fig.savefig('results/Sentences.png', dpi=300, bbox_inches='tight') 

# 5. probability measures
rs_prob, rps_prob = organize_corr_res('prob', prob_feats, split=False)

# visualize 
fig, ax = plt.subplots(1, figsize=(4, 6))
sns.heatmap(rs_prob, fmt="", annot=rps_prob,  annot_kws={"size": 12},
            cbar=True, cbar_kws={"shrink": 0.6, "orientation": 'vertical'}, 
            vmin =-vmax, vmax = vmax, 
            square=True, linewidth=.5, 
            cmap='coolwarm',
            ax=ax
            )
ax.set(xlabel="", ylabel="")
ax.xaxis.set_ticks_position('top')  
ax.yaxis.set_tick_params(rotation=0)

fig.savefig('results/prob.png', dpi=300, bbox_inches='tight') 

# 6. prompt features
rs_prompt, rps_prompt = organize_corr_res('prompt_sim', [f'Prompt_{feat}' for feat in dstr_feats + wave_feats if feat != 'MeanK2'])

# visualize 
fig, ax = plt.subplots(1, figsize=(18, 6))
sns.heatmap(rs_prompt, fmt="", annot=rps_prompt,  annot_kws={"size": 12},
            cbar=True, cbar_kws={"shrink": 0.6, "orientation": 'vertical'}, 
            vmin =-vmax, vmax = vmax, 
            square=True, linewidth=.5, 
            cmap='coolwarm',
            ax=ax
            )
ax.set(xlabel="", ylabel="")
ax.xaxis.set_ticks_position('top')  
ax.yaxis.set_tick_params(rotation=0)

fig.savefig('results/prompt_sim.png', dpi=300, bbox_inches='tight') 

# 7. graph features
rs_graph, rps_graph = organize_corr_res('graph_sim', [f'{tag}_{feat}' for tag in ['FT', 'BERT', 'Sent'] for feat in grph_feats], split=False)

# visualize 
fig, ax = plt.subplots(1, figsize=(10, 5))
sns.heatmap(rs_graph, fmt="", annot=rps_graph,  annot_kws={"size": 12},
            cbar=True, cbar_kws={"shrink": 0.6, "orientation": 'vertical'}, 
            vmin =-vmax, vmax = vmax, 
            square=True, linewidth=.5, 
            cmap='coolwarm',
            ax=ax
            )
ax.set(xlabel="", ylabel="")
ax.xaxis.set_ticks_position('top')  
ax.yaxis.set_tick_params(rotation=0)

fig.savefig('results/graph_sim.png', dpi=300, bbox_inches='tight') 
















