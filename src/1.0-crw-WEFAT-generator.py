##################################################
## Preprocesses raw data: removing stopwordss, tokenizing, etc
##################################################
## Author: Christopher White
##################################################


import sqlite3
import pandas as pd
import numpy as np
# import seaborn as sns
# from nltk.corpus import stopwords
import string
import re
# import splitter
from sklearn.metrics.pairwise import cosine_similarity
#importing the glove library
from glove import Corpus, Glove
# train/validation/test split
# from sklearn.model_selection import train_test_split
import csv

# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

db_path = '../data/processed/test.db'
# raw_path = '../data/raw/consolidated_data_raw.csv'
con = sqlite3.connect(db_path)
cur = con.cursor()

dataset = 'hate_nostop'
# dataset = 'toxic_nostop'


if dataset == 'hate_nostop':
    extract_df = pd.read_sql_query("SELECT extract from hate_clean_nostop",con)

    # print('hate full feature file')
    # print(X['extract'].head())
    print(extract_df.shape)
    #
    #
    # # separate features from label
    # y_cols = ['CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6']
    #
    # y = X.loc[:, y_cols]
    # X.drop(['id','extract','CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6'], axis=1, inplace = True)
    #
    # print('X')
    # print(X.head())
    # print('y')
    # print(y.head())

    # output_name = 'hate_clean_nostop'

else:

    extract_df = pd.read_sql_query("SELECT extract from toxic_clean_nostop", con)

    # print(X.head())
    # print(X.dtypes)
    #
    #
    # y_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    # y = X.loc[:, y_cols]
    # X.drop(['extract', "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
    #        axis=1, inplace=True)
    #
    # print(X.head())
    # print(y.head())


print(extract_df.head())


# print(df3.head())

# def split_sentences(extract_df):
#split cleansed extracts into individual tokens
compiled_output = []
for index, row in extract_df.iterrows():
    if (index % 100 == 0):
        print(index)
    # without lowercase

    text = row['extract']
    # print(text)

    # split into word tokens based on whitespace delimination
    # if text is not None:
    tokens = text.split()

    compiled_output.append(tokens)
train_cleansed_tokens = compiled_output
    # return train_cleansed_tokens


# train_cleansed_tokens = split_sentences(extract_df)

print(train_cleansed_tokens)

# print(train_cleansed_tokens[0:3])


#train glove word embedding
output_path = '../data/processed/'
file_name = 'glove.model.txt'

# https://medium.com/@japneet121/word-vectorization-using-glove-76919685ee0b
# https://textminingonline.com/getting-started-with-word2vec-and-glove-in-python
# requirements for python glove:
#     conda install llvm gcc libgcc
#     pip install glove_python

corpus = Corpus()
# training the corpus to generate the co occurence matrix which is used in GloVe
corpus.fit(train_cleansed_tokens, window=10)
# creating a Glove object which will use the matrix created in the above lines to create embeddings
# We can set the learning rate as it uses Gradient Descent and number of components
# no_components = number of dimensions
glove = Glove(no_components=300, learning_rate=0.05)

glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save(output_path + file_name)


# export dictionary to csv
df = pd.DataFrame(glove.dictionary,index=[0]).T
df.to_csv('../data/processed/glove_dictionary.csv')
df.head()

# export word embedding in format similar to Stanford Glove template
D = {word: glove.word_vectors[glove.dictionary[word]] for word in glove.dictionary.keys()}
f = open("../data/processed/dict_outfile.txt","w")
f.write( str(D) )
f.close()


# convert word embedding dict to dataframe
df_3 = pd.DataFrame.from_dict(D, orient='index')
df_3_matrix = df_3.values



# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
# scale word embedding from 0 to 1
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(df_3_matrix)
# df_3_matrix_scaled =  scaler.transform(df_3_matrix)


#take the word list from the dictionary
word_list = list(D.keys())

#set index using the word list above retrieved from the dictionary
# index is assigned without column header
df_4 = pd.DataFrame(df_3_matrix, index=[i for i in word_list])
# df_4 = pd.DataFrame(df_3_matrix_scaled, index=[i for i in word_list])
df_4.to_csv("../data/processed/word_embedding_df.txt", sep = ' ', header = False, index=True)
df_4.head(10)

# read gensim model from text file into dataframe

file ='../data/processed/word_embedding_df.txt'
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_input_file=file, word2vec_output_file="../data/processed/gensim_glove_vectors.txt")
from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format("../data/processed/gensim_glove_vectors.txt", binary=False)

ordered_vocab = [(term, voc.index, voc.count) for term, voc in model.wv.vocab.items()]
ordered_vocab = sorted(ordered_vocab, key=lambda k: k[2])
ordered_terms, term_indices, term_counts = zip(*ordered_vocab)
word_vectors = pd.DataFrame(model.wv.syn0[term_indices, :], index=ordered_terms)
word_vectors = word_vectors.reset_index(drop = False)
word_vectors = word_vectors.rename(index=str, columns={"index": "word"})
word_vectors.head()



print('hello')


# create dataframe of
pleasant = [ 'caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation']
unpleasant = ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'bomb', 'divorce', 'jail', 'poverty', 'ugly', 'cancer', 'evil', 'kill', 'rotten', 'vomit']

# python create its dataframe:
df_its1 = pd.DataFrame(pleasant,columns = ['Word'])
df_its1['Condition'] = 'Pleasant'
df_its1['Study'] = 'WEFAT1'
df_its1['Role'] = 'attribute'
df_its1 = df_its1[['Study' ,'Condition' ,'Word' ,'Role']]
# df_its1

df_its2 = pd.DataFrame(unpleasant,columns = ['Word'])
df_its2['Condition'] = 'Unpleasant'
df_its2['Study'] = 'WEFAT1'
df_its2['Role'] = 'attribute'
df_its2 = df_its2[['Study' ,'Condition' ,'Word' ,'Role']]
# df_its2

df_its = df_its1.append(df_its2)
df_its = df_its.reset_index(drop = True)
df_its.head()





#
# # reset indexes for
texts_train_idx = extract_df.reset_index(drop = False)
# texts_test_idx = texts_test.reset_index(drop = False)

# print
# print(texts_train_idx.head())

# combine test and train processed tokenized extracts
df1 = pd.DataFrame({'text': train_cleansed_tokens})
df1_idx = pd.concat([df1,texts_train_idx], axis = 1, join_axes = [df1.index])

# df2 = pd.DataFrame({'text': test_cleansed_tokens})
# df2_idx = pd.concat([df2,texts_test_idx], axis = 1, join_axes = [df2.index])
# df3 = df1_idx.append(df2_idx)
df3 = df1_idx[['index','text']]

print('df1 head:')
print(df1.head())

print('df3 head:')
print(df3.head())



#
# get vectors for items in item data
def get_vecs(its_df, study_type, number):
    vecs_df = word_vectors
    if (study_type == 'all'):
        vecs_df.set_index('word', inplace=True)
        # drop index column name
        vecs_df.index.names = [None]
        return its_df, vecs_df
    else:
        vecs_df = pd.merge(its_df, vecs_df, how='inner', left_on=['Word'], right_on=['word'])
        vecs_df = vecs_df.drop(['Study', 'Condition', 'Role', 'Word'], axis=1)
        vecs_df.set_index('word', inplace=True)
        # drop index column name
        vecs_df.index.names = [None]
        return vecs_df


def get_data(its_df, study_type, number):
    vecs_df = get_vecs(its_df, study_type, number)

    return its_df, vecs_df


def wefat(its_df, vecs_df, x_name, a_name, b_name):
    # isolate dataframes for target words, positive att, negative att
    x_words_df = its_df[its_df['Condition'] == x_name]
    a_words_df = its_df[its_df['Condition'] == a_name]
    b_words_df = its_df[its_df['Condition'] == b_name]

    # get a word list using the index of the vecs dataframe
    all_word_list = vecs_df.index.tolist()
    x_word_list = x_words_df['Word'].tolist()
    a_word_list = a_words_df['Word'].tolist()
    b_word_list = b_words_df['Word'].tolist()

    vecs_matrix = vecs_df.values

    # compute just once
    pre_cos = cosine_similarity(vecs_matrix)
    pre_cos_df = pd.DataFrame(pre_cos, columns=all_word_list, index=[i for i in all_word_list])
    pre_cos_x_a_df = pre_cos_df[pre_cos_df.index.isin(x_word_list)]

    # filter column names based on whether they actually existed in the custom word embedding
    pre_cos_x_a_df = pre_cos_x_a_df.loc[:, pre_cos_x_a_df.columns.isin(a_word_list)]
    pre_cos_x_a_mean_df = pre_cos_x_a_df.mean(axis=1, skipna=True)
    pre_cos_x_a_mean_df = pd.DataFrame(pre_cos_x_a_mean_df).T

    pre_cos_x_b_df = pre_cos_df[pre_cos_df.index.isin(x_word_list)]
    pre_cos_x_b_df = pre_cos_x_b_df.loc[:, pre_cos_x_b_df.columns.isin(b_word_list)]
    pre_cos_x_b_mean_df = pre_cos_x_b_df.mean(axis=1, skipna=True)
    pre_cos_x_b_mean_df = pd.DataFrame(pre_cos_x_b_mean_df).T

    s_xab_df = pre_cos_x_a_mean_df - pre_cos_x_b_mean_df

    pre_cos_x_denom_df = pre_cos_df[pre_cos_df.index.isin(x_word_list)]
    # use attribute words from both a and b lists
    pre_cos_x_denom_df = pre_cos_x_denom_df.loc[:, pre_cos_x_denom_df.columns.isin(a_word_list + b_word_list)]
    pre_cos_x_denom_df = pre_cos_x_denom_df.std(axis=1, skipna=True)
    pre_cos_x_denom_df = pd.DataFrame(pre_cos_x_denom_df).T

    s_wab_df = s_xab_df / pre_cos_x_denom_df

    s_wab_df = s_wab_df.T
    s_wab_df = s_wab_df.rename(index=str, columns={0: "s_wab"})

    return s_wab_df


# temp_dict = { "text" : [text],
#               "index" : [index],
# }



# apply wefat to every extract
# for i in range(2):
for i in range(df3.shape[0]):
    # for i in range(3):
    text = df3.iloc[i]['text']
    index = df3.iloc[i]['index']

    print( "text: " + str(text))
    print("index: " + str(index))

    # if (i == 0):
    #     temp_dict = { "text" : [text],
    #                   "index" : [index],
    #     }
    #
    #     df_its3 = pd.DataFrame.from_dict()


    df_its3 = pd.DataFrame(text, columns=['Word'])
    df_its3['Condition'] = 'Extract'
    df_its3['Study'] = 'WEFAT1'
    df_its3['Role'] = 'attribute'
    df_its3 = df_its3[['Study', 'Condition', 'Word', 'Role']]

    df_its3 = df_its.append(df_its3)

    its_df, vecs_df = get_data(df_its3, study_type='WEFAT', number='1')

    test = wefat(its_df,
                 vecs_df,
                 x_name="Extract",
                 a_name="Pleasant",
                 b_name="Unpleasant")

    if (i == 0):
        test_avg = pd.DataFrame(test).reset_index(drop=True).T

        # code for creating 6 max min wefats
        test_avg_6 = pd.DataFrame(test).reset_index(drop=True)
        test_avg_sorted = test_avg_6.sort_values(by=['s_wab'])
        positive_3 = test_avg_sorted.head(3).T
        positive_3.columns = ['0_top_pos_wefat', '1_top_pos_wefat', '2_top_pos_wefat']
        negative_3 = test_avg_sorted.tail(3).T
        negative_3.columns = ['0_top_neg_wefat', '1_top_neg_wefat', '2_top_neg_wefat']
        test_avg = pd.concat([test_avg, positive_3, negative_3], axis=1)

        # append index and text onto dataframe
        test_avg['index'] = index
        test_avg['text'] = str(text)

    else:
        test_avg_2 = pd.DataFrame(test).reset_index(drop=True).T

        # code for creating 6 max min wefats
        test_avg_6 = pd.DataFrame(test).reset_index(drop=True)
        test_avg_2_sorted = test_avg_6.sort_values(by=['s_wab'])
        #         print(test_avg_2_sorted)
        positive_3 = test_avg_2_sorted.head(3).T
        positive_3.columns = ['0_top_pos_wefat', '1_top_pos_wefat', '2_top_pos_wefat']
        negative_3 = test_avg_2_sorted.tail(3).T
        negative_3.columns = ['0_top_neg_wefat', '1_top_neg_wefat', '2_top_neg_wefat']
        #         print(negative_3)
        test_avg_2 = pd.concat([test_avg_2, positive_3, negative_3], axis=1)

        # append index and text onto dataframe

        test_avg_2['index'] = index
        test_avg_2['text'] = str(text)

        # append all dataframes after index 0 onto the base dataframe
        test_avg = pd.concat([test_avg, test_avg_2])

    test_avg = test_avg.reset_index(drop=True)

# replace index with true index
test_avg.set_index('index', inplace=True)
# drop index assigned column name
test_avg.index.names = [None]

# rename wefat for eveery word columns
keep_same = ['0_top_pos_wefat', '1_top_pos_wefat', '2_top_pos_wefat', '0_top_neg_wefat', '1_top_neg_wefat',
             '2_top_neg_wefat', 'text']
test_avg.columns = ['{}{}'.format(c, '' if c in keep_same else '_all_wefat') for c in test_avg.columns]

# move columns to the right
# use ix to reorder
cols = test_avg.columns.tolist()
cols_left = [x for x in cols if x not in keep_same]
cols_fin = cols_left + keep_same
# print(cols_fin)
# print(cols)
test_avg_fin = test_avg[cols_fin]
test_avg_fin.drop(['text'], axis=1)


print(test_avg_fin.head())
print(test_avg_fin.shape)

cols = test_avg_fin.columns
print('test_avg_fin cols: ' + str(cols))


db_path = '../data/processed/test.db'
# db = sqlite3.connect(db_path)
con = sqlite3.connect(db_path)
cur = con.cursor()


X = pd.read_sql_query(
    "SELECT * from hate_universal_encoder_embedding_features",
    con)
X.drop(['id'], axis=1)
print(X.shape)

cols = X.columns
print('X cols: ' + str(cols))


test_avg_fin = test_avg_fin.reset_index(drop = True)
X = X.reset_index(drop = True)
X_train_embed_wefat_df = pd.merge(X, test_avg_fin, left_index=True, right_index=True)

cols = X_train_embed_wefat_df.columns
print('X_train_embed_wefat_df cols: ' + str(cols))
print('X_train_embed_wefat_df shape: ' + str(X_train_embed_wefat_df.shape))


#shift codes to right hand side, drop some remaining unneded colums
new_cols = []
for e in cols:
    if e not in ('CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4',
       'CODE_5', 'CODE_6', 'CODE','0_top_pos_wefat', '1_top_pos_wefat', '2_top_pos_wefat',
       '0_top_neg_wefat', '1_top_neg_wefat', '2_top_neg_wefat', 'text'):
        new_cols.append(e)
cols = new_cols


code_cols = ['CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4',
       'CODE_5', 'CODE_6', 'CODE']
for e in code_cols:
    cols.append(e)

print('final cols : ' + str(cols))

X_train_embed_wefat_df = X_train_embed_wefat_df[cols]


output_name = 'hate_universal_encoder_embedding_wefat'
cur.execute("DROP TABLE IF EXISTS {}".format(output_name))
print('X_train_embed_wefat_df exported to db, table name: ' + output_name)
# X_train_embed_wefat_df.to_sql(output_name, con=con, if_exists='replace',index_label='id')
X_train_embed_wefat_df.to_sql(output_name, con=con, if_exists='replace')