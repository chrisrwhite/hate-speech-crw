# install packages in jupyter notebook
# !pip3 install tqdm
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import sqlite3
# from numba import cuda
# cuda.select_device(0)

from keras import backend as K
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = False
K.set_session(K.tf.Session(config=cfg))

db_path = '../../data/processed/test.db'
# raw_path = '../data/raw/consolidated_data_raw.csv'
db = sqlite3.connect(db_path)
# df_short = pd.read_sql_query("SELECT extract, CODE from t", db)
# df_short = pd.read_sql_query("SELECT extract, CODE from t_clean", db)
# df_short = pd.read_sql_query("SELECT extract, CODE from t_clean_nostop", db)
df_short = pd.read_sql_query("SELECT * from toxic_clean_nostop", db)

# from keras import backend as K
from numba import cuda


def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))


def embedding_fuction(df):
    tweets = list(df['comment_text'])

    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
    #     module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

    embed = hub.Module(module_url)

    tf.logging.set_verbosity(tf.logging.ERROR)

    print('generating embedding')
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        tweet_embeddings = sess.run(embed(tweets))
        tweet_embeddings_df = pd.DataFrame(tweet_embeddings)
        tweet_embeddings_columns = ['comment_text'] + tweet_embeddings_df.columns.tolist()
        tweet_embeddings_df = tweet_embeddings_df.join(df['comment_text'])
        tweet_embeddings_df = tweet_embeddings_df[tweet_embeddings_columns]

    # clear memory here:
    # https://github.com/tensorflow/tensorflow/issues/19731
    tf.reset_default_graph()

    return tweet_embeddings_df

# df = df_short.head(100)
df = df_short


start = 0
# partitions = 2000
partitions = 100
# partitions = 2
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
    #         print('if ran')
    else:
        output_df = output_df.append(tweet_embeddings_df)
    #         print('else ran')

    start += partition_size
    end += partition_size

output_df.to_csv('../../data/processed/sentence_embeddings/universal_encoder_embedding_incomplete.csv', index = False)