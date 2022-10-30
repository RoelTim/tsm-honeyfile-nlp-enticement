from spacy.lang.en.stop_words import STOP_WORDS
from itertools import chain
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric
from scipy import spatial, stats
import itertools as it, numpy as np, pandas as pd, gensim.corpora as corpora
import json, statistics, pprint, string, re, gensim, os, itertools, spacy, os, time, glob #docx
#from sklearn import preprocessing
import matplotlib.patches as mpatches
#from sklearn.manifold import TSNE
from collections import Counter
from tqdm import tqdm

def topics():
    path_this_file = os.path.dirname(os.path.abspath(__file__))    
    master_df_path = os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/master_df.csv'))
    df = pd.read_csv(master_df_path)
    local_context_file_path = os.path.abspath(os.path.join(path_this_file,'..', 'data/raw/local_context'))
    print('\nlocal_context_file_path: \t', local_context_file_path)
    local_files = glob.glob(local_context_file_path + '/**/*.docx', recursive=True)
    print('number of local context files: \t', len(local_files))
    nlp = spacy.load("en_core_web_sm")

    class EntityRetokenizeComponent:
        """ merges tokens that are entities """
        def __init__(self, nlp):
            pass
        def __call__(self, doc):
            with doc.retokenize() as retokenizer:
                for ent in doc.ents:
                    retokenizer.merge(doc[ent.start:ent.end],
                                      attrs = {"LEMMA": str(doc[ent.start:ent.end])})
            return doc

    nlp.add_pipe(EntityRetokenizeComponent(nlp))

    for nr_files_per_local_context in tqdm([5, 10, 20]):
        categories = ['computer', 'plants', 'theater', 'customs']
        local_context_files = {}

        def divide_chunks(l, n): 
            for i in range(0, len(l), n):  
                yield l[i:i + n] 

        for cat in categories:
            x = list(set(df.loc[df["category_local_context_file"]==cat, 'local_context_file']))
            y = list(divide_chunks(x, nr_files_per_local_context))
            if len(y[-1]) < nr_files_per_local_context:
                y = y[:-1]
            local_context_files[cat] = y

        with open(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/preprocessed_text.txt'))) as file:
            preprocessed_text = json.load(file)

        ##############################################################################
        ######################### Top N words for Local Context ######################
        ##############################################################################

        nr_of_top_words=50
        all_top_words = {}
        for cat, local_files in local_context_files.items():
            top_words =[]
            for selection_local_files in local_files:
                local_context_text = []
                for j in selection_local_files:
                    for key, value in preprocessed_text.items():
                        if str(j) in str(key) and '.lda' in str(key):
                            local_context_text.append(value)
                local_context_text = [item for sublist in local_context_text for item in sublist]
                local_context_text = [item for sublist in local_context_text for item in sublist]
                counter = {}
                for i in local_context_text: counter[i] = counter.get(i, 0) + 1
                sort_dict = sorted([ (freq,word) for word, freq in counter.items() ], reverse=True)
                sorted_words = [item[1] for item in sort_dict]
                most_common_words = sorted_words[0:nr_of_top_words]# int(len(sorted_words) * .05)]
                top_words.append(most_common_words)
            all_top_words[cat]=top_words

        with open(str(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/'))) + '/local_context_top_'+str(nr_of_top_words)+'_local_context_size_of_' + str(nr_files_per_local_context) + 'dict_', 'w') as f:
            json.dump(all_top_words, f)

        ##############################################################################      
        #################### LDA topic words for Local Context #######################
        ##############################################################################

        topics_of_local_context = {}

        for cat, local_files in local_context_files.items():
            all_topics =[]
            for selection_local_files in local_files:
                local_context_text = []
                for j in selection_local_files:
                    for key, value in preprocessed_text.items():
                        if str(j) in str(key) and '.lda' in str(key):
                            local_context_text.append(value)

                local_context_text = [item for sublist in local_context_text for item in sublist]

                # Make Bigram     
                bigram = gensim.models.Phrases(local_context_text) # higher threshold fewer phrases.
                bigram_mod = gensim.models.phrases.Phraser(bigram)
                local_context_text=bigram_mod[local_context_text]

                id2word = corpora.Dictionary(local_context_text) # Create Dictionary
                corpus = [id2word.doc2bow(b) for b in local_context_text] # Term Document Frequency

                # Base Model 
                lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                              id2word=id2word,
                                              num_topics=5, 
                                              random_state=100,
                                              chunksize=100,
                                              passes=10,
                                              per_word_topics=True)

                # Get a list of all the keywords of the topics we found
                topics_strings = []
                for m in lda_model.show_topics():
                    for k in m:
                        if len(str(k)) > 2:
                            topics_strings.append(re.findall('"([^"]*)"',str(k)))

                topics = [item for elem in topics_strings for item in elem]  
                all_topics.append(topics)
            topics_of_local_context[cat]=all_topics

        with open(str(os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/'))) + '/local_context_topics_dict_'+str(nr_files_per_local_context), 'w') as f:
            json.dump(topics_of_local_context, f)

if __name__ == "__main__":
    topics()


