##################################################
## Encodes the preprocessed data into sentence emeddings
## 1. Universal encoder
##################################################
## Author: Christopher White
##################################################


# install packages in jupyter notebook
# !pip3 install tqdm



# from tqdm import tqdm


import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import sqlite3


# from numba import cuda
# cuda.select_device(0)

# from keras import backend as K
# cfg = K.tf.ConfigProto()
# cfg.gpu_options.allow_growth = False
# K.set_session(K.tf.Session(config=cfg))


db_path = '../data/processed/test.db'
# db = sqlite3.connect(db_path)
con = sqlite3.connect(db_path)
cur = con.cursor()


dataset = 'hate_nostop'
# dataset = 'toxic_nostop'


# df_short = pd.read_sql_query("SELECT extract, CODE from t", db)
# df_short = pd.read_sql_query("SELECT extract, CODE from t_clean", db)

if dataset == 'hate_nostop':
    print('hate')
    partitions = 10
    df_short = pd.read_sql_query("SELECT extract, CODE_0, CODE_1, CODE_2, CODE_3, CODE_4, CODE_5, CODE_6, CODE from hate_clean_nostop", con)
    output_file = '../data/processed/sentence_embeddings/hate_universal_encoder_embedding_features.csv'
    output_name = 'hate_universal_encoder_embedding_features'

else:
    print('toxic')
    partitions = 100

    df_short = pd.read_sql_query("SELECT extract, toxic, severe_toxic, obscene, threat, insult, identity_hate from toxic_clean_nostop", con)
    output_file = '../data/processed/sentence_embeddings/toxic_universal_encoder_embedding_features.csv'
    output_name = 'toxic_universal_encoder_embedding_features'

# generate universal sentence embedding using this function
# improvement=> save model to local drive
# https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15

def embedding_fuction(df):
    tweets = list(df['extract'])

    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

    embed = hub.Module(module_url)

    tf.logging.set_verbosity(tf.logging.ERROR)

    print('generating embedding')
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        tweet_embeddings = sess.run(embed(tweets))
        tweet_embeddings_df = pd.DataFrame(tweet_embeddings)
        tweet_embeddings_columns = ['extract'] + tweet_embeddings_df.columns.tolist()
        tweet_embeddings_df = tweet_embeddings_df.join(df['extract'])
        tweet_embeddings_df = tweet_embeddings_df[tweet_embeddings_columns]

    # clear memory here:
    # https://github.com/tensorflow/tensorflow/issues/19731
    tf.reset_default_graph()

    return tweet_embeddings_df

df = df_short


start = 0
# partitions = 100
partition_size = int(df.shape[0] / partitions)
remainder = df.shape[0] - (partitions * partition_size)

end = partition_size
for partition in range(partitions):
    #for the first partition, add the remainder onto the size, since there is an odd number of records, ~71 leftover
    if (partition == 0) :
        print('remainder = ' +str(remainder))
        end += remainder

    if (partition == 1):
        end -= remainder

    print('partition number: ' + str(partition))
    #     print('start ' + str(start))
    #     print('end ' + str(end))

    temp_df = df.ix[start: end - 1]
    print('temp_df size ' + str(temp_df.shape))
    tweet_embeddings_df = embedding_fuction(temp_df)

    #     print('partition ' + str(partition))
    if partition == 0:
        output_df = tweet_embeddings_df
    else:
        output_df = output_df.append(tweet_embeddings_df)

    start += partition_size
    end += partition_size

df_short = df_short.reset_index(drop=False)
output_df = output_df.reset_index(drop=False)

print(df_short.head(20))
print(output_df.head(20))



print('df_short.shape: ' + str(df_short.shape))
print('output_df.shape: ' + str(output_df.shape))


# merge on the labels onto the feature file
df_short.drop('extract', axis=1, inplace=True)

print(output_df.dtypes)
print(df_short.dtypes)

# complete_output_df = pd.merge(output_df,df_short, left_index=True, right_index=True)
complete_output_df = pd.merge(output_df, df_short, how='left', on=['index', 'index'])
complete_output_df.drop('index', axis=1, inplace=True)

# complete_output_df = output_df.join(df_short)

# complete_output_df = pd.concat([output_df, df_short], axis=1)

print('complete_output_df.head()')
print(complete_output_df.head())

complete_output_df.to_csv(output_file, index = False)

print('cleansed text exported to db, table name: ' + output_name)
complete_output_df.to_sql(output_name, con=con, if_exists='replace',index_label='id')