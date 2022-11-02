""" functions that help to generate the heatmaps plots of the paper """

import pandas as pd
import numpy as np

def frame(ind, axs):
    """add a black frame around all the heatmaps"""
    if ind is None:
        for axis in ['top', 'bottom', 'left', 'right']:
            axs.spines[axis].set_visible(True)
            axs.spines[axis].set_color('black')
            axs.spines[axis].set_visible(True)
            axs.spines[axis].set_color('black')
    else:
        for axis in ['top', 'bottom', 'left', 'right']:
            axs[ind].spines[axis].set_visible(True)
            axs[ind].spines[axis].set_color('black')
            axs[ind].spines[axis].set_visible(True)
            axs[ind].spines[axis].set_color('black')

def score_dataframe(score, topics, dec_paths):
    """ create a dataframe that contains the TSM enticement scores """
    categories_vector = []
    for key, value in topics.items():
        categories_vector.extend([key] * len(value))
    score_df = pd.DataFrame(score, index=categories_vector, columns=dec_paths)
    score_df["categories_local_context"] = categories_vector
    score_df = score_df.melt(id_vars=["categories_local_context"], \
                             var_name=["dec_paths"], \
                             value_name='score')
    for cat in ['customs', 'plants', 'computer', 'theater']:
        score_df.loc[score_df['dec_paths'].str.contains(cat), \
                     'category_deception_file'] = cat
    score_df["method"] = np.nan
    score_df.loc[score_df['dec_paths'].str.contains('lorem'), 'method'] = 'lorem'
    for gen in ['pos_words', 'dependency_parsed_tokens', 'gpt']:
        score_df.loc[score_df['dec_paths'].str.contains(gen), 'method'] = 'normal'
    return score_df

def select_data(gen_method, score, topics, dec_paths, \
                categories, percentile, tsm_method, \
                local_context_n, perc, ind, matrix):
    """ select and organize data based on many input parameters """
    score_df = score_dataframe(score, topics, dec_paths)
    score_df['score'] = (score_df['score'] - min(score_df['score'])) / \
                    (max(score_df['score']) - min(score_df['score']))
    selection1 = score_df[(score_df['method'] == 'normal')]
    for cat1 in categories:
        for cat2 in categories:
            selection2 = selection1[(selection1['category_deception_file'] == cat1)\
                                    &(selection1["categories_local_context"] == cat2)]
            matrix[ind, 0] = gen_method
            matrix[ind, 1] = cat1
            matrix[ind, 2] = cat2
            matrix[ind, 3] = percentile
            matrix[ind, 4] = np.percentile(selection2['score'], percentile)
            matrix[ind, 5] = tsm_method
            matrix[ind, 6] = local_context_n
            matrix[ind, 7] = perc
            ind += 1
            del selection2
    del selection1
    return matrix, ind

def data_df(matrix, variation):
    """ organise dataframe """
    df_data = pd.DataFrame(matrix, columns=['deception file generation method', \
                               'deception file category', \
                               'local context category', \
                               'percentile', \
                               'value', \
                               'scoring measure', \
                               'nr. of files per local context', \
                               variation])
    df_data = df_data.drop_duplicates()
    df_data['value'] = df_data['value'].astype(float)
    return df_data
