import os, glob, 
, docx, json
# Preprocess Deception and Local Context Files with Spacy
# remove stop words, punctuation, numbers
# apply lemmatisation
# apply named-entity recognition (NER)

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

def preprocess_deception_files():
    print('start preprocessing deception files')
    path_this_file = os.path.dirname(os.path.abspath(__file__))    
    path_deception_files = os.path.abspath(os.path.join(path_this_file,'..', 'data/raw/honeyfiles'))
    dec_files = glob.glob(path_deception_files + '/**/*.docx', recursive=True)
    #output_file = os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/preprocessed_text.txt'))
    nlp = spacy.load("en_core_web_sm") #!python -m spacy download en_core_web_sm, we use the "en_core_web_sm" model of Spacy, because there is (currently) a bug in "en_core_web_lg" to select stop words.
    nlp.add_pipe(EntityRetokenizeComponent(nlp))
    decep_files = {}
    for dec_file in dec_files:
        doc = docx.Document(dec_file)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        decep_files[dec_file] = '\n'.join(fullText) 

    doc2vec_text = {}
    lda_text = {}
    sbmtm_text = {}
    preprocessed_text = {}

    for count, (i, j) in enumerate(decep_files.items()):
        if count % 100 == 0: print(count, ' of ', len(decep_files))
        j = j.lower()
        tokens = nlp(j)
        processed_text = []
        for ar in tokens.sents:
            ar = str(ar)
            x = nlp(ar)
            art = []
            for token in x:
                if token.pos_ != 'PUNCT' and token.pos_ != 'NUM':
                    if token.is_stop == False and len(token)>2 and '\n' not in token.lemma_: #remove stopwords short characters
                        art.append(str(token.lemma_)) #lemmatisation
            processed_text.append(art)
        sbmtm = [item for sublist in processed_text for item in sublist]

        doc2vec_text[i] = ' '.join(sbmtm)
        lda_text[i] = processed_text
        sbmtm_text[i] = processed_text 
        if 'theater' in j:
            category = 'theater'
        elif 'plants' in j:
            category = 'plants'
        elif 'computer' in j:
            category = 'computer'
        elif 'customs' in j or 'abfnotices' in j:
            category = 'customs'
        else:
            category = 'unknown'
        metadata = i + '...' + category + '...N/A...dec...doc2vec'
        preprocessed_text[metadata] = ' '.join(sbmtm)
        metadata = i + '...' + category + '...N/A...dec...lda'
        preprocessed_text[metadata] = processed_text
    output_file = os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/preprocessed_text.txt'))
    json.dump(preprocessed_text, open(output_file,'w'))
    print('finished preprocessing deception files')
          
def preproces_local_context_files():  
    print('start preprocessing local context files')
    #cwd = os.getcwd()
    
    path_this_file = os.path.dirname(os.path.abspath(__file__))    
    path_local_files = os.path.abspath(os.path.join(path_this_file,'..', 'data/raw/local_context'))
    local_files = glob.glob(path_local_files + '/**/*.docx', recursive=True)
    print(len(local_files))
    output_file = os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/preprocessed_text.txt'))
    
    nlp = spacy.load("en_core_web_sm") #!python -m spacy donwload en_core_web_sm, we use the "en_core_web_sm" model of Spacy, because there is (currently) a bug in "en_core_web_lg" to select stop words.
    nlp.add_pipe(EntityRetokenizeComponent(nlp))

    with open(output_file) as file:
        preprocessed_text = json.load(file)
        
    for count, i in enumerate(local_files):
        if count % 100 == 0: print(count, ' of ', len(local_files))
        try: 
            doc = docx.Document(i)
            fullText = []
            for para in doc.paragraphs:
                fullText.append(para.text)
            j = '\n'.join(fullText)        
            j = j.lower()
            tokens = nlp(j)
            processed_text = []
            for ar in tokens.sents:
                ar = str(ar)
                x = nlp(ar)
                art = []
                for token in x:
                    if token.pos_ != 'PUNCT' and token.pos_ != 'NUM':
                        if token.is_stop == False and len(token)>2 and '\n' not in token.lemma_: #remove stopwords short characters
                            art.append(str(token.lemma_)) #lemmatisation
                processed_text.append(art)
            sbmtm = [item for sublist in processed_text for item in sublist]
            if 'theater' in i:
                category = 'theater'
            elif 'plants' in i:
                category = 'plants'
            elif 'computer' in i:
                category = 'computer'
            elif 'customs' in i or 'abfnotices' in i:
                category = 'customs'
            else:
                category = 'unknown'
            metadata = i + '...' + category + '...1...local...doc2vec'
            preprocessed_text[metadata] = ' '.join(sbmtm)
            metadata = i + '...' + category + '...1...local...lda'
            preprocessed_text[metadata] = processed_text
        except: 
            print('skip: ', i)
    
    output_file = os.path.abspath(os.path.join(path_this_file,'..', 'data/processed/preprocessed_text.txt'))
    json.dump(preprocessed_text, open(output_file, 'w'))
    print('finished preprocessing local context files')    

if __name__ == "__main__":
    preprocess_deception_files()
    preproces_local_context_files()