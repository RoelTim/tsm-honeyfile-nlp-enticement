"""
Extract Topics from Local Context files: (1) Top N words & (2) LDA topic words.
Local context size varies: 5, 10, 20
"""
import os
import glob
import itertools
import re
import json
import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
import spacy

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

    return df_comb

def divide_chunks(full_list, nr_chunks):
    """ divide a list in nr_chunks chunks """
    for i in range(0, len(full_list), nr_chunks):
        yield full_list[i:i + nr_chunks]

class EntityRetokenizeComponent:
    """ merges tokens that are entities """
    def __init__(self, nlp):
        pass
    def __call__(self, doc):
        with doc.retokenize() as retokenizer:
            for ent in doc.ents:
                retokenizer.merge(doc[ent.start:ent.end],
                                  attrs={"LEMMA": str(doc[ent.start:ent.end])})
        return doc

def top_n_words(nr_of_top_words, \
                            local_context_files, \
                            preprocessed_text):
    """
    Top N words for Local Context
    """
    all_top_words = {}
    for cat, local_files in local_context_files.items():
        top_words = []
        for selection_local_files in local_files:
            local_context_text = []
            for j in selection_local_files:
                for key, value in preprocessed_text.items():
                    if str(j) in str(key) and '.lda' in str(key):
                        local_context_text.append(value)
            local_context_text = [item for sublist in local_context_text for item in sublist]
            local_context_text = [item for sublist in local_context_text for item in sublist]
            counter = {}
            for i in local_context_text:
                counter[i] = counter.get(i, 0) + 1
            sort_dict = sorted([(freq, word) for word, \
                                freq in counter.items()], reverse=True)
            sorted_words = [item[1] for item in sort_dict]
            most_common_words = sorted_words[0:nr_of_top_words]
            top_words.append(most_common_words)
        all_top_words[cat] = top_words
        return all_top_words

def lda_topic_model(local_context_files, preprocessed_text):
    """ LDA topic words for Local Context """
    topics_of_local_context = {}
    for cat, local_files in local_context_files.items():
        all_topics = []
        for selection_local_files in local_files:
            local_context_text = []
            for j in selection_local_files:
                for key, value in preprocessed_text.items():
                    if str(j) in str(key) and '.lda' in str(key):
                        local_context_text.append(value)

            local_context_text = [item for sublist in local_context_text for item in sublist]

            # Make Bigram
            bigram = gensim.models.Phrases(local_context_text) #higher threshold fewer phrases.
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            local_context_text = bigram_mod[local_context_text]
            id2word = gensim.corpora.Dictionary(local_context_text) # Create Dictionary
            corpus = [id2word.doc2bow(b) for b in local_context_text] # Term Document Frequency

            # Base Model
            lda_model = gensim.models.LdaMulticore(corpus=corpus, \
                                                   id2word=id2word, \
                                                   num_topics=5, \
                                                   random_state=100, \
                                                   chunksize=100, \
                                                   passes=10, \
                                                   per_word_topics=True)

            # Get a list of all the keywords of the topics we found
            topics_strings = []
            for lda_topics in lda_model.show_topics():
                for single_topic in lda_topics:
                    if len(str(single_topic)) > 2:
                        topics_strings.append(re.findall('"([^"]*)"', \
                                                         str(single_topic)))
            topics = [item for elem in topics_strings for item in elem]
            all_topics.append(topics)
        topics_of_local_context[cat] = all_topics
    return topics_of_local_context

def get_topics():
    """ Extract Topics from Local Context files:
    (1) Top N words & (2) LDA topic words.
    Local context size varies: 5, 10, 20
    Args: None
    Returns: dic: Dictionary with topics.
    """
    path_this_file = os.path.dirname(os.path.abspath(__file__))
    df_comb = categorise()

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(EntityRetokenizeComponent(nlp))

    for nr_files_per_local_context in tqdm([5, 10, 20]):
        categories = ['computer', 'plants', 'theater', 'customs']
        local_context_files = {}

        for cat in categories:
            list_cat = list(set(df_comb.loc[df_comb["category_local_context_file"] == cat, \
                                       'local_context_file']))
            list_chunks = list(divide_chunks(list_cat, nr_files_per_local_context))
            if len(list_chunks[-1]) < nr_files_per_local_context:
                list_chunks = list_chunks[:-1]
            local_context_files[cat] = list_chunks

        with open(os.path.abspath(os.path.join(path_this_file, \
                                               '..', \
                                               'data/processed/preprocessed_text.txt'))) \
        as file:
            preprocessed_text = json.load(file)

        nr_of_top_words = 50
        all_top_words = top_n_words(nr_of_top_words, \
                                    local_context_files, \
                                    preprocessed_text)

        with open(str(os.path.abspath(os.path.join(path_this_file, \
                                                   '..', \
                                                   'data/processed/'))) \
                  + '/local_context_top_' + \
                  str(nr_of_top_words) + \
                  '_local_context_size_of_' + \
                  str(nr_files_per_local_context) + \
                  'dict_', 'w') as file:
            json.dump(all_top_words, file)

        topics_of_local_context = lda_topic_model(local_context_files, \
                                                  preprocessed_text)

        with open(str(os.path.abspath(os.path.join(path_this_file, \
                                                   '..', \
                                                   'data/processed/'))) \
                  + '/local_context_topics_dict_' + \
                  str(nr_files_per_local_context), 'w') as file:
            json.dump(topics_of_local_context, file)

if __name__ == "__main__":
    get_topics()
