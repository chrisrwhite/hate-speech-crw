##################################################
## Preprocesses raw data: removing stopwordss, tokenizing, etc
##################################################
## Author: Christopher White
##################################################


import sqlite3
import pandas as pd
import numpy as np
# import seaborn as sns
from nltk.corpus import stopwords
import string
import re
# import splitter
# from sklearn.metrics.pairwise import cosine_similarity
#importing the glove library
# from glove import Corpus, Glove
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


# df_short = pd.read_sql_query("SELECT extract, CODE from t", db)
# df_short = pd.read_sql_query("SELECT extract, CODE from t_clean", db)





if dataset == 'hate_nostop':
    df_short = pd.read_sql_query("SELECT extract, CODE_0, CODE_1, CODE_2, CODE_3, CODE_4, CODE_5, CODE_6 from hate", con)

    # separate features from label

    data_df = df_short.loc[:, ['extract']]
    label_df = df_short.loc[:, [ 'CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6']]

    output_name = 'hate_clean_nostop'

else:
    df_short = pd.read_sql_query("SELECT extract,toxic,severe_toxic,obscene,threat,insult,identity_hate from toxic", con)

    # separate features from label

    data_df = df_short.loc[:, ['extract']]
    label_df = df_short.loc[:, ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

    output_name = 'toxic_clean_nostop'

# df_short = pd.read_sql_query("SELECT extract, CODE from t_clean", db)

# see how much memory the dataset uses
print(df_short.memory_usage(deep=True).sum())


# cleanse and tokenize extracts

def process_data(df):
    # removed the negative stop words based on:
    #     https://www.cs.cmu.edu/~ark/EMNLP-2015/proceedings/WASSA/pdf/WASSA14.pdf

    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                  "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                  'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                  'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                  'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                  'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                  'to', 'from', 'up', 'down', 'in', 'out', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                  'there', 'when', 'where', 'why', 'how', 'any', 'both', 'each', 'other', 'some', 'such', 'only', 'own',
                  'same', 'so', 'than', 'too', 's', 't', 'can', 'will', 'just', 'now', 'd', 'll', 'm', 'o', 're', 've',
                  'y', 'ma']

    compiled_output = []
    for index, row in df.iterrows():
        if (index % 100 == 0):
            print(index)
        # without lowercase

        no_url_txt = row['extract']
        # no_url_txt = row['comment_text']

        # confirm in the science paper, because in glove, capitals make a difference: use exact case for names

        # remove timestamps
        no_url_txt = re.sub(r'\d+:\d+', ' ', no_url_txt)

        # split into word tokens based on whitespace delimination
        tokens = no_url_txt.split()

        # remove remainint tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]

        # remove stop words (first time)
        tokens = [word for word in tokens if word not in stop_words]

        consolidated_output = tokens

        #       SPLIT COMPOUND WORDS
        #         consolidated_output = []
        #         for item in tokens:
        # #             split_items = segment_str(item, exclude=None)

        #             split_items = splitter.split(item)
        #             consolidated_output.extend(split_items)

        # remove stop words (second time)
        consolidated_output = [word for word in consolidated_output if word not in stop_words]

        consolidated_output = " ".join(consolidated_output)
        #         print(consolidated_output)

        compiled_output.append(consolidated_output)

    return compiled_output


train_cleansed_tokens = process_data(data_df)

# print(train_cleansed_tokens)


# df_new = pd.DataFrame(train_cleansed_tokens, columns = ['extract'])
df_new = pd.DataFrame(train_cleansed_tokens, columns = ['extract'])
df_new = df_new.join(label_df)

print('df_new.head()')
print(df_new.tail())

cur.execute("DROP TABLE IF EXISTS {}".format(output_name))
print('cleansed text exported to db, table name: ' + output_name)
df_new.to_sql(output_name, con=con, if_exists='replace',index_label='id')

# df_new['comment_text'].to_csv('./plain_extracts_' + output_name + '.txt' ,index = False, header = False)