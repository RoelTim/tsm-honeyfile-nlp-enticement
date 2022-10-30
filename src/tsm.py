#calculate the Topic Semantic Matching Enticement Score

import itertools as it, numpy as np, pandas as pd
import os, spacy, json, ast, glob, pprint, warnings,collections #h5py
from itertools import product
import matplotlib.pyplot as plt #seaborn as sns,
from scipy.sparse import *
from numpy import savez_compressed
from gensim.models import CoherenceModel, doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy import spatial, stats
from collections import OrderedDict
#from sklearn import preprocessing
from tqdm import tqdm

def wordvec_matrix(words, type_words):
    if type_words == 'topic':
        m = np.zeros((100, 300)) #features = 300
    else:
        m = np.zeros((4500, 300))
    for idx, w in enumerate(words): m[idx] = w.vector/np.linalg.norm(w.vector) if w.has_vector else np.NaN
    m = np.float32(m) #NEW TO SAVE SPACE 13/09/2021
    return m

class EntityRetokenizeComponent: # merges tokens that are entities
    def __init__(self, nlp):
        pass
    def __call__(self, doc):
        with doc.retokenize() as retokenizer:
            for ent in doc.ents:
                retokenizer.merge(doc[ent.start:ent.end],
                                  attrs = {"LEMMA": str(doc[ent.start:ent.end])})
        return doc

def prep_deception_files():
    path_this_file = os.path.dirname(os.path.abspath(__file__))    
    df_path = os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/master_df.csv'))
    preprocessed_files = os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/preprocessed_text.txt'))
    path_output = os.path.abspath(os.path.join(path_this_file,'..', 'data/processed'))
    df = pd.read_csv(df_path) #enticement/data/processed
    nlp = spacy.load("en_core_web_lg") #only lg model has word vectors #!python -m spacy download en_core_web_lg
    nlp.add_pipe(EntityRetokenizeComponent(nlp))

    with open(preprocessed_files) as file:
        preprocessed_text = json.load(file)  
    dec_paths=np.unique(list(df['deception_file'])) #list of paths to deception files
    
    deceptions2 = []
    for deception_text in dec_paths:
        for key, value in preprocessed_text.items():
            if str(deception_text) in str(key) and '.lda' in str(key):
                deceptions2.append(' '.join(list([c for lst in value for c in lst])))
                break
                
    dec_tokenised = list(nlp.pipe(deceptions2, disable=["tagger", "parser", 'textcat'])) #generator
    dec_matrix = [wordvec_matrix(dec, 'words') for dec in dec_tokenised]    
    dec_word_count = [np.count_nonzero(i[:,0]) for i in dec_matrix]#get the length of all the 658 deception files

    np.save(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_texts')), deceptions2, allow_pickle=True)           
    #np.save(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_tokenised')), dec_tokenised, allow_pickle=True)
    np.save(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_paths')), dec_paths, allow_pickle=True)
    np.save(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_matrix')), dec_matrix, allow_pickle=True)
    np.save(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_word_count')), dec_word_count, allow_pickle=True)

def tsm_lda():
    """Calculate the TSM enticement score with the LDA topic model"""
    print('load spacy')
    nlp = spacy.load("en_core_web_lg") #only lg model has word vectors #!python -m spacy download en_core_web_lg
    nlp.add_pipe(EntityRetokenizeComponent(nlp))
    print('loading spacy done')
    
    path_this_file = os.path.dirname(os.path.abspath(__file__))     
    dec_texts = np.load(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_texts'))+'.npy',allow_pickle=True)
    dec_texts = [str(i) for i in list(dec_texts)]
    dec_tokenised = list(nlp.pipe(dec_texts, disable=["tagger", "parser", 'textcat'])) #generator
    dec_paths = np.load(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_paths'))+'.npy',allow_pickle=True)
    dec_matrix = np.load(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_matrix'))+'.npy',allow_pickle=True)
    dec_word_count = np.load(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_word_count'))+'.npy',allow_pickle=True)
    
    path_output = os.path.abspath(os.path.join(path_this_file,'..', 'data/processed'))
    nr_files_per_local_context = [5, 10, 20]
    thresholds = [0.3, 0.6, 0.9]
    for n in nr_files_per_local_context:
        print('nr of files', n, ' of [5, 10, 20]')
        with open(path_output+'/local_context_topics_dict_' +str(n), 'r') as f:
            topics = json.load(f)
        topics = OrderedDict(sorted(topics.items())) 
        nr_topics = [len(item) for sublist in list(topics.values()) for item in sublist]
        topics1 = [item for sublist in list(topics.values()) for item in sublist]
        topics1 = [list(i) for i in topics1]
        topics2 = [' '.join(topic).replace('_',' ') for topic in topics1]        
        topics3 = list(nlp.pipe(topics2, disable=["tagger", "parser", 'textcat']))
        topics_ms = [wordvec_matrix(top, 'topic') for top in topics3]
        for t in thresholds:
            print('thresholds:', t, ' of ', len(thresholds))
            lda_score_average = np.empty((len(topics1), len(dec_tokenised)))
            lda_score_average[:, :] = np.NaN
            lda_score_with_threshold = np.empty((len(topics1), len(dec_tokenised)))
            lda_score_with_threshold[:, :] = np.NaN
            for i, (m, nr_topic) in enumerate(tqdm(zip(topics_ms,nr_topics))):
                x = np.float32(m @ np.concatenate(dec_matrix).T)
                x = np.where(x!=0, x, np.nan)
                y = (np.array(np.hsplit(x, len(dec_tokenised))) +1 )/2
                lda_score_average[i,:] = np.nanmean(np.nanmean(y, axis=1),axis=1)
                z = np.where(y>t, y, 0) / nr_topic
                lda_score_with_threshold[i,:] = np.nansum(np.nansum(z, axis=1),axis=1)/dec_word_count
                if np.amax(lda_score_with_threshold[i,:]) > 1:
                    sys.exit('error')
                del x, y,z
            np.save(path_output+'/lda_score_average_'+str(n)+'.npy', lda_score_average)
            np.save(path_output+'/lda_score_with_threshold_'+str(t)+"_"+str(n)+'.npy', lda_score_with_threshold)
    
def tsm_percentages():
    """Calculate the TSM enticement score with the topic percentages words"""
    
    path_this_file = os.path.dirname(os.path.abspath(__file__))     

    print('load spacy')
    nlp = spacy.load("en_core_web_lg") #only lg model has word vectors #!python -m spacy download en_core_web_lg
    nlp.add_pipe(EntityRetokenizeComponent(nlp))
    print('loading spacy done')    

    dec_texts = np.load(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_texts'))+'.npy',allow_pickle=True)
    dec_texts = [str(i) for i in list(dec_texts)]
    dec_tokenised = list(nlp.pipe(dec_texts, disable=["tagger", "parser", 'textcat'])) #generator
    dec_matrix = np.load(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_matrix'))+'.npy',allow_pickle=True)
    dec_word_count = np.load(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_word_count'))+'.npy',allow_pickle=True)
    path_output = os.path.abspath(os.path.join(path_this_file,'..', 'data/processed'))
    nlp = spacy.load("en_core_web_lg") #only lg model has word vectors #!python -m spacy download en_core_web_lg
    nlp.add_pipe(EntityRetokenizeComponent(nlp))

    percentages = [0.5, 1, 5]
    for perc in percentages:
        print('percentage', perc, ' of [0.5, 1, 5]')
        with open(path_output+'/local_context_topics_dict_10', 'r') as f:
            topics = json.load(f, object_pairs_hook=OrderedDict)
        topics = OrderedDict(sorted(topics.items()))
        nr_topics = [len(item) for sublist in list(topics.values()) for item in sublist]
        topics1 = [item for sublist in list(topics.values()) for item in sublist]
        topics1 = [list(i) for i in topics1]
        topics2 = [' '.join(topic).replace('_',' ') for topic in topics1]        
        topics3 = list(nlp.pipe(topics2, disable=["tagger", "parser", 'textcat']))
        topics_ms = [wordvec_matrix(top, 'topic') for top in topics3]
        lda_score_with_percentage = np.empty((len(topics1), len(dec_tokenised)))
        lda_score_with_percentage[:, :] = np.NaN
        for i, (m, nr_topic) in enumerate(zip(topics_ms, nr_topics)):
            if i % 50 == 0: print(i, ' of ', len(topics_ms))
            x = np.float32(m @ np.concatenate(dec_matrix).T)
            x = np.where(x!=0, x, np.nan)
            y = (np.array(np.hsplit(x, len(dec_tokenised)))+1)/2
            z = np.where(y >= np.nanpercentile(y, (100-perc)), y, np.nan) / nr_topic
            lda_score_with_percentage[i,:] = np.nansum(np.nansum(z, axis=1),axis=1)/dec_word_count
        np.save(path_output+'/lda_score_top_percent_'+str(perc)+'_10.npy', lda_score_with_percentage)     

def tsm_sbmtm():
    """Calculate the TSM enticement score with the SBM Topic Model"""
    path_this_file = os.path.dirname(os.path.abspath(__file__))  
    
    print('load spacy')
    nlp = spacy.load("en_core_web_lg") #only lg model has word vectors #!python -m spacy download en_core_web_lg
    nlp.add_pipe(EntityRetokenizeComponent(nlp))
    print('loading spacy done')    
        
    #dec_tokenised = np.load(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_tokenised'))+'.npy',allow_pickle=True)
    dec_texts = np.load(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_texts'))+'.npy',allow_pickle=True)
    dec_texts = [str(i) for i in list(dec_texts)]
    dec_tokenised = list(nlp.pipe(dec_texts, disable=["tagger", "parser", 'textcat'])) #generator
    
    dec_matrix = np.load(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_matrix'))+'.npy',allow_pickle=True)
    dec_matrix = np.float32(dec_matrix)
    dec_word_count = np.load(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_word_count'))+'.npy',allow_pickle=True)
    dec_word_count = dec_word_count.astype(np.int32)
    
    path_output = os.path.abspath(os.path.join(path_this_file,'..', 'data/processed'))          
    nlp = spacy.load("en_core_web_lg") #only lg model has word vectors #!python -m spacy download en_core_web_lg
    nlp.add_pipe(EntityRetokenizeComponent(nlp))
    
    with open(path_output+'/SMBTM_local_context_topics_dict_10', 'r') as f:
        SBMTM_topics = json.load(f, object_pairs_hook=OrderedDict)
    SBMTM_topics = OrderedDict(sorted(SBMTM_topics.items())) 
    
    SBMTM_topics_list = []
    for keys, values in SBMTM_topics.items():
        for i in values:
            topics1 = []
            topics1_value = []
            for key, value in i.items():
                topics1.append([j[0] for j in value])        
                topics1_value.append([j[1] for j in value])    
            topics1 = [j for i in topics1 for j in i]
            topics1_value = [j for i in topics1_value for j in i]
            index = np.argpartition(topics1_value, -50)[-50:] #index top 50 topics
            SBMTM_topics_list.append(' '.join(list(np.array(topics1)[index])))   
    
    topics3_sbmtm = list(nlp.pipe(SBMTM_topics_list, disable=["tagger", "parser", 'textcat']))
    del SBMTM_topics, SBMTM_topics_list
    topics_sbmtm_ms = [wordvec_matrix(t, 'topic') for t in topics3_sbmtm]

    score_sbmtm_average = np.empty((len(topics3_sbmtm), len(dec_tokenised)))
    score_sbmtm_average[:, :] = np.NaN
    score_sbmtm_with_threshold = np.empty((len(topics3_sbmtm), len(dec_tokenised)))

    del topics3_sbmtm
    score_sbmtm_with_threshold[:, :] = np.NaN
    
    for i, m in enumerate(topics_sbmtm_ms):
        if i % 50 == 0: print(i, ' of ', len(topics_sbmtm_ms))
        x = np.float32(m @ np.concatenate(dec_matrix).T)
        x = np.where(x!=0, x, np.nan)
        y = (np.array(np.hsplit(x, len(dec_tokenised))) + 1) / 2 #(658, 68, 2869)
        del x, m
        sc1 = np.nanmean(np.nanmean(y, axis=1),axis=1)
        sc1[np.isnan(sc1)] = 0
        score_sbmtm_average[i,:] = sc1
        del sc1
        t=0.9
        z = np.where(y>t, y, np.nan) / 50
        sc2 = np.nansum(np.nansum(z, axis=1),axis=1)/dec_word_count
        del z, y
        sc2[np.isnan(sc2)] = 0
        score_sbmtm_with_threshold[i,:] = sc2 
        del sc2
    np.save(path_output+'/score_sbmtm_average_10.npy', score_sbmtm_average)
    np.save(path_output+'/score_sbmtm_with_threshold_0.9_10.npy', score_sbmtm_with_threshold)

def tsm_most_frequent_words():
    
    path_this_file = os.path.dirname(os.path.abspath(__file__))    
    
    print('load spacy')
    nlp = spacy.load("en_core_web_lg") #only lg model has word vectors #!python -m spacy download en_core_web_lg
    nlp.add_pipe(EntityRetokenizeComponent(nlp))
    print('loading spacy done')    
        
    #dec_tokenised = np.load(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_tokenised'))+'.npy',allow_pickle=True)
    dec_texts = np.load(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_texts'))+'.npy',allow_pickle=True)
    dec_texts = [str(i) for i in list(dec_texts)]
    dec_tokenised = list(nlp.pipe(dec_texts, disable=["tagger", "parser", 'textcat'])) #generator
      
    dec_matrix = np.load(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_matrix'))+'.npy',allow_pickle=True)
    dec_word_count = np.load(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/dec_word_count'))+'.npy',allow_pickle=True)
    path_output = os.path.abspath(os.path.join(path_this_file,'..', 'data/processed'))
    nlp = spacy.load("en_core_web_lg") #only lg model has word vectors #!python -m spacy download en_core_web_lg
    nlp.add_pipe(EntityRetokenizeComponent(nlp))
    
    #print(type(dec_tokenised))
    #print(type(dec_tokenised[0]))
    #print(type(dec_matrix))
    #print(type(dec_matrix[0]))
    #print(type(dec_word_count))
    #print(type(dec_word_count[0]))
    
    nr_of_top_words = [50]
    nr_files_per_local_context = [10]
    thresholds = [0.9]
    for freq in nr_of_top_words:
        for n in nr_files_per_local_context:
            print('n: ', n)
            with open(path_output+'/local_context_top_'+str(freq)+'_local_context_size_of_10dict_', 'r') as f:
                most_frequent_words = json.load(f, object_pairs_hook=OrderedDict)
                
            print(len(most_frequent_words))
            most_frequent_words = OrderedDict(sorted(most_frequent_words.items()))
            print(len(most_frequent_words))

            topics1 = [item for sublist in list(most_frequent_words.values()) for item in sublist]
            topics1 = [list(i) for i in topics1]
            print(len(topics1))
            topics2 = [' '.join(topic).replace('_',' ') for topic in topics1]
            print(len(topics1))
            topics3 = list(nlp.pipe(topics2, disable=["tagger", "parser", 'textcat']))
            print(len(topics3))
         
            topics_ms = [wordvec_matrix(top, 'topic') for top in topics3]
            for t in thresholds:
                print('t: ', t)
                most_frequent_words_average = np.empty((len(topics1), len(dec_tokenised)))
                most_frequent_words_average[:, :] = np.NaN
                most_frequent_words_threshold = np.empty((len(topics1), len(dec_tokenised)))
                most_frequent_words_threshold[:, :] = np.NaN
                for i, m in enumerate(topics_ms):
                    if i % 50 ==0: print(i, ' of ', len(topics_ms))
                    x = np.float32(m @ np.concatenate(dec_matrix).T)
                    x = np.where(x!=0, x, np.nan)
                    y = (np.array(np.hsplit(x, len(dec_tokenised))) +1 )/2
                    most_frequent_words_average[i,:] = np.nanmean(np.nanmean(y, axis=1),axis=1)
                    z = np.where(y>t, y, 0) / nr_of_top_words
                    most_frequent_words_threshold[i,:] = np.nansum(np.nansum(z, axis=1),axis=1)/dec_word_count
                    del x, y,z
                np.save(path_output+'/most_frequent_words_average_10_' + str(freq) +'.npy', most_frequent_words_average)
                np.save(path_output+'/most_frequent_words_threshold_'+str(t)+ "_" + '10_' +str(freq)+'.npy', most_frequent_words_threshold)
                
if __name__ == "__main__":
    #prep_deception_files() #ok 28/10/2022
    #tsm_lda() #ok 29/10/2022
    #tsm_percentages() #ok 29/10/2022
    #tsm_sbmtm() #ok 29/10/202
    tsm_most_frequent_words()