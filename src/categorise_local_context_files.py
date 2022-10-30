#import matplotlib.pyplot as plt
#from spacy.lang.en.stop_words import STOP_WORDS
from itertools import chain 
#from PyDictionary import PyDictionary #https://www.synonym.com/
#from spacy.lang.en.stop_words import STOP_WORDS
#from gensim.utils import simple_preprocess
#from gensim.models import CoherenceModel, doc2vec
#from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric
#from scipy import spatial, stats
#from nltk.tokenize import word_tokenize 
import itertools as it, numpy as np, pandas as pd#, gensim.corpora as corpora
import json, pprint, string, re, gensim, os, itertools, glob #docx
#from nltk.tokenize import word_tokenize
#from nltk.corpus import wordnet
#from sklearn import preprocessing

def categorise():

    print("let's start")

    path_this_file = os.path.dirname(os.path.abspath(__file__))    

    print("path_this_file:", path_this_file)

    #local_context_file_path = os.path.join(cwd, 'local_context_enticement')

    local_context_file_path = os.path.abspath(os.path.join(path_this_file,'..', 'data/raw/local_context'))
    print("local_context_file_path:", local_context_file_path)
    local_files = glob.glob(local_context_file_path + '/**/*.docx', recursive=True)
    print("local_files:", local_files)

    deception_file_path = os.path.abspath(os.path.join(path_this_file,'..', 'data/raw/honeyfiles'))
    print('deception_file_path:', deception_file_path)
    dec_files = glob.glob(deception_file_path + '/**/*.docx', recursive=True)
    print('dec_files:', dec_files)

    #create all combinations between the deception files and the local context files
    combinations = list(itertools.product(dec_files, local_files))
    pprint.pprint(combinations[0:3])
    df = pd.DataFrame(data=combinations, columns=['deception_file','local_context_file'])
    #df["category_deception_file"] = np.nan
    df["category_local_context_file"] = np.nan

    #categorise data: plants, customs, computer and theater
    #df.loc[df['deception_file'].str.contains('customs'), 'category_deception_file'] = 'customs'
    #df.loc[df['deception_file'].str.contains('ABF'), 'category_deception_file'] = 'customs'
    #df.loc[df['deception_file'].str.contains('abf'), 'category_deception_file'] = 'customs'
    #df.loc[df['deception_file'].str.contains('\d{4}-\d{2}'), "category_deception_file"] = 'customs'
    #df.loc[df['deception_file'].str.contains('plants'), 'category_deception_file'] = 'plants'
    #df.loc[df['deception_file'].str.contains('wild-useful-herbs-of-aktobe-region-western'), 'category_deception_file'] = 'plants'
    #df.loc[df['deception_file'].str.contains('computer'), 'category_deception_file'] = 'computer'
    #df.loc[df['deception_file'].str.contains('08552374'), 'category_deception_file'] = 'computer'
    #df.loc[df['deception_file'].str.contains('theater'), 'category_deception_file'] = 'theater'
    #df.loc[df['deception_file'].str.contains('0021989420918654'), 'category_deception_file'] = 'theater'

    df.loc[df['local_context_file'].str.contains('customs'), 'category_local_context_file'] = 'customs'
    df.loc[df['local_context_file'].str.contains('ABF'), 'category_local_context_file'] = 'customs'
    df.loc[df['local_context_file'].str.contains('abf'), 'category_local_context_file'] = 'customs'
    df.loc[df['local_context_file'].str.contains('plants'), 'category_local_context_file'] = 'plants'
    df.loc[df['local_context_file'].str.contains('computer'), 'category_local_context_file'] = 'computer'
    df.loc[df['local_context_file'].str.contains('theater'), 'category_local_context_file'] = 'theater'

    #remove all the templates and tests
    df = df[~df.deception_file.str.contains("template")]
    df = df[~df.deception_file.str.contains("test")]
    df = df[~df.deception_file.str.contains("I_fucking_can_fly")]
    df = df[~df.local_context_file.str.contains("template")]
        
    dfname = os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/master_df.csv'))
    output_file = open(dfname, 'wb')
    df.to_csv(dfname, index = False)
    output_file.close()

if __name__ == "__main__":
    categorise()





