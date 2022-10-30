import pandas as pd, numpy as np

def frame(ind, axs):
    """add a black frame around all the heatmaps"""
    if ind == None:
        for axis in ['top','bottom','left','right']:
            axs.spines[axis].set_visible(True)
            axs.spines[axis].set_color('black')
            axs.spines[axis].set_visible(True)
            axs.spines[axis].set_color('black') 
    else:
        for axis in ['top','bottom','left','right']:
            axs[ind].spines[axis].set_visible(True)
            axs[ind].spines[axis].set_color('black')
            axs[ind].spines[axis].set_visible(True)
            axs[ind].spines[axis].set_color('black')
            
def score_dataframe(score, topics, dec_paths):
    categories_vector = []
    for key, value in topics.items():
        categories_vector.extend([key]*len(value))
    score_df = pd.DataFrame(score, index = categories_vector, columns = dec_paths)
    score_df["categories_local_context"] = categories_vector
    score_df = score_df.melt(id_vars=["categories_local_context"], var_name=["dec_paths"], value_name = 'score')
    score_df.loc[score_df['dec_paths'].str.contains('customs|ABF|abf|\d{4}-\d{2}'), 'category_deception_file'] = 'customs'
    score_df.loc[score_df['dec_paths'].str.contains('plants|wild-useful-herbs-of-aktobe-region-western'), 'category_deception_file'] = 'plants'
    score_df.loc[score_df['dec_paths'].str.contains('computer|08552374'), 'category_deception_file'] = 'computer'
    score_df.loc[score_df['dec_paths'].str.contains('theater|0021989420918654'), 'category_deception_file'] = 'theater'
    score_df["method"] = np.nan
    score_df.loc[score_df['dec_paths'].str.contains('lorem'), 'method'] = 'lorem'
    score_df.loc[score_df['dec_paths'].str.contains('pos_words'), 'method'] = 'normal'
    score_df.loc[score_df['dec_paths'].str.contains('dependency_parsed_tokens'), 'method'] = 'normal'
    score_df.loc[score_df['dec_paths'].str.contains('gpt_variant2'), 'method'] = 'normal'
    score_df.loc[score_df['dec_paths'].str.contains('gpt_variant1'), 'method'] = 'normal'
    score_df.loc[score_df['dec_paths'].str.contains('unknown'), 'method'] = 'unknown'
    score_df = score_df[~score_df.dec_paths.str.contains("unknown")]    
    return score_df
                 
def select_data(gen_method, score, topics, dec_paths, categories, p, sm, n, perc, i, matrix):
    score_df = score_dataframe(score, topics, dec_paths)
    score_df['score'] = (score_df['score']-min(score_df['score']))/(max(score_df['score'])-min(score_df['score']))
    selection1 = score_df[(score_df['method']=='normal')]
    for cat1 in categories:
        for cat2 in categories:
            selection2 = selection1[(selection1['category_deception_file']==cat1)&(selection1["categories_local_context"]==cat2)]
            matrix[i,0]= gen_method
            matrix[i,1]=cat1
            matrix[i,2]=cat2
            matrix[i,3]=p
            matrix[i,4]=np.percentile(selection2['score'],  p)
            matrix[i,5]=sm
            matrix[i,6]=n
            matrix[i,7]=perc
            i+=1   
            del selection2
    del selection1
    return matrix,i

def data_df(matrix, variation):
    df = pd.DataFrame(matrix, columns = ['deception file generation method',
                               'deception file category',
                               'local context category',
                               'percentile',
                               'value', 
                               'scoring measure', 
                               'nr. of files per local context', 
                               variation])
    df=df.drop_duplicates()
    df['value'] = df['value'].astype(float)
    return df
        
        
        