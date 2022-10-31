""" Calculate the Topic Semantic Matching Enticement Score """
import os
import glob
import json
import itertools
from collections import OrderedDict
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

def categorise():
    """
    Classify the local context file as plants, computer, customs or theater
    And make all the combinations of honeyfiles and local context files
    """
    path_this_file = os.path.dirname(os.path.abspath(__file__))
    local_context_file_path = os.path.abspath(os.path.join(path_this_file, \
                                                           '..', \
                                                           'data/raw/local_context'))
    local_files = glob.glob(local_context_file_path + '/**/*.docx', recursive=True)
    deception_file_path = os.path.abspath(os.path.join(path_this_file, '..', 'data/raw/honeyfiles'))
    dec_files = glob.glob(deception_file_path + '/**/*.docx', recursive=True)

    combinations = list(itertools.product(dec_files, local_files))
    df_comb = pd.DataFrame(data=combinations, columns=['deception_file', 'local_context_file'])
    df_comb["category_local_context_file"] = np.nan
    df_comb.loc[df_comb['local_context_file'].str.contains('customs'), \
           'category_local_context_file'] = 'customs'
    df_comb.loc[df_comb['local_context_file'].str.contains('ABF'), \
           'category_local_context_file'] = 'customs'
    df_comb.loc[df_comb['local_context_file'].str.contains('abf'), \
           'category_local_context_file'] = 'customs'
    df_comb.loc[df_comb['local_context_file'].str.contains('plants'), \
           'category_local_context_file'] = 'plants'
    df_comb.loc[df_comb['local_context_file'].str.contains('computer'), \
           'category_local_context_file'] = 'computer'
    df_comb.loc[df_comb['local_context_file'].str.contains('theater'), \
           'category_local_context_file'] = 'theater'

    #remove all the templates and tests
    df_comb = df_comb[~df_comb.deception_file.str.contains("template")]
    df_comb = df_comb[~df_comb.deception_file.str.contains("test")]
    df_comb = df_comb[~df_comb.deception_file.str.contains("I_fucking_can_fly")]
    df_comb = df_comb[~df_comb.local_context_file.str.contains("template")]
    return df_comb

def wordvec_matrix(words, type_words):
    """ create word vector matrix """
    if type_words == 'topic':
        mat = np.zeros((100, 300))
    else:
        mat = np.zeros((4500, 300))
    for idx, word in enumerate(words):
        mat[idx] = word.vector/np.linalg.norm(word.vector) if word.has_vector else np.NaN
    mat = np.float32(mat)
    return mat

class EntityRetokenizeComponent:
    """ entity recognition """
    def __init__(self, nlp):
        pass
    def __call__(self, doc):
        with doc.retokenize() as retokenizer:
            for ent in doc.ents:
                retokenizer.merge(doc[ent.start:ent.end],
                                  attrs={"LEMMA": str(doc[ent.start:ent.end])})
        return doc

def prep_deception_files():
    """ prepare the deception files """
    path_this_file = os.path.dirname(os.path.abspath(__file__))
    preprocessed_files = os.path.abspath(os.path.join(path_this_file, \
                                                      '..', \
                                                      'data/processed/preprocessed_text.txt'))

    df_comb = categorise()
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe(EntityRetokenizeComponent(nlp))
    with open(preprocessed_files) as file:
        preprocessed_text = json.load(file)
    dec_paths = np.unique(list(df_comb['deception_file']))
    deceptions2 = []
    for deception_text in dec_paths:
        for key, value in preprocessed_text.items():
            if str(deception_text) in str(key) and '.lda' in str(key):
                deceptions2.append(' '.join(list([c for lst in value for c in lst])))
                break
    dec_tokenised = list(nlp.pipe(deceptions2, disable=["tagger", "parser", 'textcat']))
    dec_matrix = [wordvec_matrix(dec, 'words') for dec in dec_tokenised]
    dec_word_count = [np.count_nonzero(i[:, 0]) for i in dec_matrix]
    np.save(os.path.abspath(os.path.join(path_this_file, \
                                         '..', \
                                         'data/processed/dec_texts')), \
            deceptions2, \
            allow_pickle=True)
    np.save(os.path.abspath(os.path.join(path_this_file, \
                                         '..', \
                                         'data/processed/dec_paths')), \
            dec_paths, \
            allow_pickle=True)
    np.save(os.path.abspath(os.path.join(path_this_file, \
                                         '..', \
                                         'data/processed/dec_word_count')), \
            dec_word_count, \
            allow_pickle=True)

def tsm_lda():
    """Calculate the TSM enticement score with the LDA topic model"""
    print('load spacy')
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe(EntityRetokenizeComponent(nlp))
    print('loading spacy done')

    path_this_file = os.path.dirname(os.path.abspath(__file__))
    dec_texts = np.load(os.path.abspath(os.path.join(path_this_file, \
                                                     '..', \
                                                     'data/processed/dec_texts')) +\
                        '.npy', \
                        allow_pickle=True)
    dec_texts = [str(i) for i in list(dec_texts)]
    dec_tokenised = list(nlp.pipe(dec_texts, disable=["tagger", \
                                                      "parser", \
                                                      'textcat']))

    dec_matrix = [wordvec_matrix(dec, 'words') for dec in dec_tokenised]
    dec_word_count = np.load(os.path.abspath(os.path.join(path_this_file, \
                                                          '..', \
              'data/processed/dec_word_count')) \
                             + '.npy', \
                             allow_pickle=True)

    path_output = os.path.abspath(os.path.join(path_this_file, \
                                               '..', \
                                               'data/processed'))
    nr_files_per_local_context = [5, 10, 20]
    thresholds = [0.3, 0.6, 0.9]
    for nr_file in nr_files_per_local_context:
        print('nr of files', nr_file, ' of [5, 10, 20]')
        with open(path_output+'/local_context_topics_dict_' + \
                  str(nr_file), 'r') as file:
            topics = json.load(file)
        topics = OrderedDict(sorted(topics.items()))
        nr_topics = [len(item) for sublist in list(topics.values()) for item in sublist]
        topics1 = [item for sublist in list(topics.values()) for item in sublist]
        topics1 = [list(i) for i in topics1]
        topics2 = [' '.join(topic).replace('_', ' ') for topic in topics1]
        topics3 = list(nlp.pipe(topics2, disable=["tagger", "parser", 'textcat']))
        topics_ms = [wordvec_matrix(top, 'topic') for top in topics3]
        for thres in thresholds:
            print('thresholds:', thres, ' of [0.3, 0.6, 0.9]')
            lda_score_average = np.empty((len(topics1), len(dec_tokenised)))
            lda_score_average[:, :] = np.NaN
            lda_score_with_threshold = np.empty((len(topics1), len(dec_tokenised)))
            lda_score_with_threshold[:, :] = np.NaN
            for i, (topics, nr_topic) in enumerate(tqdm(zip(topics_ms, nr_topics))):
                matr_prod = np.float32(topics @ np.concatenate(dec_matrix).T)
                matr_prod = np.where(matr_prod != 0, matr_prod, np.nan)
                matr_prod_scale = (np.array(np.hsplit(matr_prod, len(dec_tokenised))) + 1) / 2
                lda_score_average[i, :] = np.nanmean(np.nanmean(matr_prod_scale, axis=1), axis=1)
                matr_prod_scale_thres = np.where(matr_prod_scale > \
                                                 thres, matr_prod_scale, 0) \
                / nr_topic
                lda_score_with_threshold[i, :] = \
                np.nansum(np.nansum(matr_prod_scale_thres, \
                                    axis=1), \
                          axis=1) / dec_word_count
                del matr_prod, matr_prod_scale, matr_prod_scale_thres
            np.save(path_output + '/lda_score_average_' + \
                    str(nr_file) + '.npy', lda_score_average)
            np.save(path_output + '/lda_score_with_threshold_' + \
                    str(thres) + "_" + str(nr_file) + '.npy', \
                    lda_score_with_threshold)

def tsm_percentages():
    """Calculate the TSM enticement score with the topic percentages words"""
    path_this_file = os.path.dirname(os.path.abspath(__file__))
    print('load spacy')
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe(EntityRetokenizeComponent(nlp))
    print('loading spacy done')
    dec_texts = np.load(os.path.abspath(os.path.join(path_this_file, \
                                                     '..' \
                                                     , 'data/processed/dec_texts')) \
                        + '.npy', \
                        allow_pickle=True)
    dec_texts = [str(i) for i in list(dec_texts)]
    dec_tokenised = list(nlp.pipe(dec_texts, disable=["tagger", "parser", 'textcat']))
    dec_matrix = [wordvec_matrix(dec, 'words') for dec in dec_tokenised]
    dec_word_count = np.load(os.path.abspath(os.path.join(path_this_file, \
                                                          '..', \
                                                          'data/processed/dec_word_count')) + \
                             '.npy', \
                             allow_pickle=True)
    path_output = os.path.abspath(os.path.join(path_this_file, \
                                               '..', \
                                               'data/processed'))
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe(EntityRetokenizeComponent(nlp))

    percentages = [0.5, 1, 5]
    for perc in percentages:
        print('percentage', perc, ' of [0.5, 1, 5]')
        with open(path_output+'/local_context_topics_dict_10', 'r') as file:
            topics = json.load(file, object_pairs_hook=OrderedDict)
        topics = OrderedDict(sorted(topics.items()))
        nr_topics = [len(item) for sublist in list(topics.values()) for item in sublist]
        topics1 = [item for sublist in list(topics.values()) for item in sublist]
        topics1 = [list(i) for i in topics1]
        topics2 = [' '.join(topic).replace('_', ' ') for topic in topics1]
        topics3 = list(nlp.pipe(topics2, disable=["tagger", "parser", 'textcat']))
        topics_ms = [wordvec_matrix(top, 'topic') for top in topics3]
        lda_score_with_percentage = np.empty((len(topics1), len(dec_tokenised)))
        lda_score_with_percentage[:, :] = np.NaN
        for i, (topics, nr_topic) in enumerate(zip(topics_ms, nr_topics)):
            if i % 50 == 0:
                print(i, ' of ', len(topics_ms))
            matr_prod = np.float32(topics @ np.concatenate(dec_matrix).T)
            matr_prod = np.where(matr_prod != 0, matr_prod, np.nan)
            matr_prod_scale = (np.array(np.hsplit(matr_prod, \
                                                  len(dec_tokenised))) + \
                               1) / 2
            matr_prod_scale_thres = np.where(matr_prod_scale \
                                             >= np.nanpercentile(matr_prod_scale, \
                                                                 (100-perc)), \
                         matr_prod_scale, np.nan) / nr_topic
            lda_score_with_percentage[i, :] = np.nansum(np.nansum(matr_prod_scale_thres, axis=1), \
                                                        axis=1) / dec_word_count
        np.save(path_output + '/lda_score_top_percent_' + \
                str(perc) + '_10.npy', lda_score_with_percentage)

def tsm_sbmtm():
    """Calculate the TSM enticement score with the SBM Topic Model"""
    path_this_file = os.path.dirname(os.path.abspath(__file__))
    print('load spacy')
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe(EntityRetokenizeComponent(nlp))
    print('loading spacy done')
    dec_texts = np.load(os.path.abspath(os.path.join(path_this_file, \
                                                     '..', \
                                                     'data/processed/dec_texts')) +\
                        '.npy', \
                        allow_pickle=True)
    dec_texts = [str(i) for i in list(dec_texts)]
    dec_tokenised = list(nlp.pipe(dec_texts, disable=["tagger", \
                                                      "parser", \
                                                      'textcat']))
    dec_matrix = [wordvec_matrix(dec, 'words') for dec in dec_tokenised]
    dec_word_count = np.load(os.path.abspath(os.path.join(path_this_file, \
                                                          '..', \
                                                          'data/processed/dec_word_count')) + \
                             '.npy', \
                             allow_pickle=True)
    dec_word_count = dec_word_count.astype(np.int32)
    path_output = os.path.abspath(os.path.join(path_this_file, '..', 'data/processed'))
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe(EntityRetokenizeComponent(nlp))
    with open(path_output + '/SMBTM_local_context_topics_dict_10', 'r') as file:
        sbmtm_topics = json.load(file, object_pairs_hook=OrderedDict)
    sbmtm_topics = OrderedDict(sorted(sbmtm_topics.items()))
    sbmtm_topics_list = []
    for values in sbmtm_topics.values():
        for i in values:
            topics1 = []
            topics1_value = []
            for value in i.values():
                topics1.append([j[0] for j in value])
                topics1_value.append([j[1] for j in value])
            topics1 = [j for i in topics1 for j in i]
            topics1_value = [j for i in topics1_value for j in i]
            index = np.argpartition(topics1_value, -50)[-50:]
            sbmtm_topics_list.append(' '.join(list(np.array(topics1)[index])))
    topics3_sbmtm = list(nlp.pipe(sbmtm_topics_list, disable=["tagger", \
                                                              "parser", \
                                                              'textcat']))
    del sbmtm_topics, sbmtm_topics_list
    topics_sbmtm_ms = [wordvec_matrix(t, 'topic') for t in topics3_sbmtm]

    score_sbmtm_average = np.empty((len(topics3_sbmtm), len(dec_tokenised)))
    score_sbmtm_average[:, :] = np.NaN
    score_sbmtm_with_threshold = np.empty((len(topics3_sbmtm), len(dec_tokenised)))

    del topics3_sbmtm
    score_sbmtm_with_threshold[:, :] = np.NaN
    for i, topics in enumerate(topics_sbmtm_ms):
        if i % 50 == 0:
            print(i, ' of ', len(topics_sbmtm_ms))
        matr_prod = np.float32(topics @ np.concatenate(dec_matrix).T)
        matr_prod = np.where(matr_prod != 0, matr_prod, np.nan)
        matr_prod_scale = (np.array(np.hsplit(matr_prod, len(dec_tokenised))) + 1) / 2
        del matr_prod, topics
        sc1 = np.nanmean(np.nanmean(matr_prod_scale, axis=1), axis=1)
        sc1[np.isnan(sc1)] = 0
        score_sbmtm_average[i, :] = sc1
        del sc1
        thres = 0.9
        matr_prod_scale_thres = np.where(matr_prod_scale > thres, matr_prod_scale, np.nan) / 50
        sc2 = np.nansum(np.nansum(matr_prod_scale_thres, axis=1), axis=1) / dec_word_count
        del matr_prod_scale_thres, matr_prod_scale
        sc2[np.isnan(sc2)] = 0
        score_sbmtm_with_threshold[i, :] = sc2
        del sc2
    np.save(path_output + '/score_sbmtm_average_10.npy', score_sbmtm_average)
    np.save(path_output + '/score_sbmtm_with_threshold_0.9_10.npy', \
            score_sbmtm_with_threshold)

def tsm_most_frequent_words():
    """ tsm enticement score based on most frequent words """
    path_this_file = os.path.dirname(os.path.abspath(__file__))
    print('load spacy')
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe(EntityRetokenizeComponent(nlp))
    print('loading spacy done')
    dec_texts = np.load(os.path.abspath(os.path.join(path_this_file, \
                                                     '..', \
                                                     'data/processed/dec_texts')) \
                        + '.npy', \
                        allow_pickle=True)
    dec_texts = [str(i) for i in list(dec_texts)]
    dec_tokenised = list(nlp.pipe(dec_texts, disable=["tagger", \
                                                      "parser", \
                                                      'textcat']))
    dec_matrix = [wordvec_matrix(dec, 'words') for dec in dec_tokenised]
    dec_word_count = np.load(os.path.abspath(os.path.join(path_this_file, \
                                                          '..', \
                             'data/processed/dec_word_count')) + \
                             '.npy', \
                             allow_pickle=True)
    path_output = os.path.abspath(os.path.join(path_this_file, \
                                               '..', \
                                               'data/processed'))
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe(EntityRetokenizeComponent(nlp))
    freq = 50
    thres = 0.9
    with open(path_output + '/local_context_top_' + \
              str(freq) + '_local_context_size_of_10dict_', 'r') as file:
        most_frequent_words = json.load(file, object_pairs_hook=OrderedDict)
    most_frequent_words = OrderedDict(sorted(most_frequent_words.items()))
    topics1 = [item for sublist in list(most_frequent_words.values()) \
               for item in sublist]
    topics1 = [list(i) for i in topics1]
    topics2 = [' '.join(topic).replace('_', ' ') for topic in topics1]
    topics3 = list(nlp.pipe(topics2, disable=["tagger", \
                                              "parser", \
                                              'textcat']))
    topics_ms = [wordvec_matrix(top, 'topic') for top in topics3]
    most_frequent_words_average = np.empty((len(topics1), len(dec_tokenised)))
    most_frequent_words_average[:, :] = np.NaN
    most_frequent_words_threshold = np.empty((len(topics1), len(dec_tokenised)))
    most_frequent_words_threshold[:, :] = np.NaN
    for i, topics in enumerate(topics_ms):
        if i % 50 == 0:
            print(i, ' of ', len(topics_ms))
        matr_prod = np.float32(topics @ np.concatenate(dec_matrix).T)
        matr_prod = np.where(matr_prod != 0, matr_prod, np.nan)
        matr_prod_scale = (np.array(np.hsplit(matr_prod, len(dec_tokenised))) + 1) / 2
        most_frequent_words_average[i, :] = np.nanmean(np.nanmean(matr_prod_scale, axis=1), axis=1)
        matr_prod_scale_thres = np.where(matr_prod_scale > thres, matr_prod_scale, 0) / freq
        most_frequent_words_threshold[i, :] = \
        np.nansum(np.nansum(matr_prod_scale_thres, axis=1), \
                  axis=1)/dec_word_count
        del matr_prod, matr_prod_scale, matr_prod_scale_thres
    np.save(path_output + '/most_frequent_words_average_10_' + \
            str(freq) + '.npy', most_frequent_words_average)
    np.save(path_output + '/most_frequent_words_threshold_' + \
            str(thres) +  "_" + '10_' + str(freq) + '.npy', \
            most_frequent_words_threshold)

if __name__ == "__main__":
    prep_deception_files()
    tsm_lda()
    tsm_percentages()
    tsm_sbmtm()
    tsm_most_frequent_words()
