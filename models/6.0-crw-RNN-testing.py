import pandas as pd
import sqlite3

#need to update input file to ESN Clean _Nostop, on my desktop

# path_input = "../data/processed/sentence_embeddings/infersent_feature_file_without_stopwords_clean.csv"

# path_input = "../data/processed/sentence_embeddings/tox_infersent_feature_file_without_stopwords_clean.csv"

path_input = "../data/processed/sentence_embeddings/universal_feature_file_without_stopwords.csv"



db_path = '../data/processed/test.db'
# db = sqlite3.connect(db_path)
con = sqlite3.connect(db_path)
cur = con.cursor()



# dataset = 'hate_nostop'
dataset = 'toxic_nostop'




if dataset == 'hate_nostop':
    X = pd.read_sql_query(
        "SELECT * from hate_universal_encoder_embedding_features",
        con)

    print('hate full feature file')
    print(X['extract'].head())
    print(X.shape)


    # separate features from label
    y_cols = ['CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6']

    y = X.loc[:, y_cols]
    X.drop(['id','extract','CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6'], axis=1, inplace = True)

    print('X')
    print(X.head())
    print('y')
    print(y.head())

    # output_name = 'hate_clean_nostop'

else:

    X = pd.read_sql_query("SELECT * from toxic_universal_encoder_embedding_features", con)

    print(X.head())
    print(X.dtypes)


    y_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = X.loc[:, y_cols]
    X.drop(['extract', "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
           axis=1, inplace=True)

    print(X.head())
    print(y.head())


    # output_name = 'toxic_clean_nostop'


print('creating train and test sets')

import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_np = np.array(X_train)
X_test_np = np.array(X_test)
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# convert numpy stuff back into pandas dataframes
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train, columns = y_cols)


X_test = pd.DataFrame(X_test_np)
y_test = pd.DataFrame(y_test_np, columns = y_cols)

print(y_test.head())



# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
# The three dimensions of LSTM input are:

# 1. Samples. One sequence is one sample. A batch is comprised of one or more samples.
# 2. Time Steps. One time step is one point of observation in the sample.
# 3. Features. One feature is one observation at a time step.

# batch_size = 1

import numpy as np

def reshape_data(X_train, X_test, y_train, y_test):
    X_train_reshape = np.reshape(X_train.values, ( X_train.shape[0],1, X_train.shape[1]))

    X_test_reshape = np.reshape(X_test.values, ( X_test.shape[0],1, X_test.shape[1]))

    y_train_reshape_matrix = y_train.values
    y_train_reshape_matrix = y_train_reshape_matrix.reshape(y_train.shape[0],1,y_train.shape[1])

    y_test_reshape_matrix = y_test.values
    y_test_reshape_matrix = y_test_reshape_matrix.reshape(y_test.shape[0],1,y_test.shape[1])


    # print(X_train.shape)
    print(X_train_reshape.shape)

    # print(X_test.shape)
    print(X_test_reshape.shape)

    # print(y_train.shape)
    print(y_train_reshape_matrix.shape)

    return X_train_reshape, y_train_reshape_matrix,X_test_reshape, y_test_reshape_matrix


X_train_reshape, y_train_reshape_matrix,X_test_reshape, y_test_reshape_matrix = reshape_data(X_train, X_test, y_train, y_test)

from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import make_pipeline

from string import punctuation
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator




from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Dense, Flatten, LSTM, MaxPooling1D, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard

#LSTM
def simple_LSTM(X_train_reshape, y_train_reshape_matrix,X_test_reshape, y_test_reshape_matrix,vector_size,model_batch_size,no_epochs):



    # model = Sequential()

    data_dim = X_train_reshape.shape[2]
    timesteps = X_train_reshape.shape[1]


    # use variable length timesteps using None
    # https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras
    # data_dim = X_train_reshape.shape[2]
    timesteps = None


    # We usually match up the size of the embedding layer output with the number of hidden layers in the LSTM cell.
    # https://adventuresinmachinelearning.com/keras-lstm-tutorial/
    # hidden_size = 4096
    # hidden_size = 32
    hidden_size = 512

    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=True, input_shape=(timesteps, data_dim)))
    model.add(LSTM(hidden_size, return_sequences=True))
    # model.add(Dense(7, activation='linear'))
    # model.add(Dense(6, activation='sigmoid'))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    print(model.summary())

    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


    # Keras detects the output_shape and automatically determines which accuracy to use when accuracy is specified. For multi-class classification, categorical_accuracy will be used internally.
    # https://stackoverflow.com/questions/43544358/categorical-crossentropy-need-to-use-categorical-accuracy-or-accuracy-as-the-met

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['categorical_accuracy'])

    tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, write_graph=True, write_images=True)


    print('data_dim shape: ' + str(data_dim))
    print('X_train_reshape shape: ' + str(X_train_reshape.shape))
    print('y_train_reshape_matrix shape: ' + str(y_train_reshape_matrix.shape))


    # model.fit(X_train_reshape,y_train_reshape_matrix, batch_size=500, epochs=15)
    model.fit(X_train_reshape, y_train_reshape_matrix, batch_size=model_batch_size, epochs=no_epochs,
             validation_data=(X_test_reshape, y_test_reshape_matrix), callbacks=[tensorboard, EarlyStopping(min_delta=0.0001, patience=3)])


    print('generate final prediction accuracy metrics')
    val_lost, val_acc = model.evaluate(X_test_reshape,y_test_reshape_matrix)

    print('final val_lost ' + str(val_lost))
    print('final val_acc ' + str(val_acc))

    return val_lost, val_acc


vector_size = 512
model_batch_size = 500
no_epochs = 25

# run LSTM
# val_lost, val_acc = simple_LSTM(X_train_reshape, y_train_reshape_matrix,X_test_reshape, y_test_reshape_matrix,vector_size,model_batch_size,no_epochs)

# result list to store accuracies across all folds ...  probably should output
# consolidated dataframe of all processed records

model_count = 0
# for index, row in df_files.iterrows():
#
#     feature_file_df = pd.read_csv(row['paths'])
#     #     testing feature files
#     #     feature_file_df = pd.read_csv(row['paths'], nrows=20)
#     #     print(feature_file_df.head())
#     X = feature_file_df
#     #     print(X.head())
#     #     X = X.drop(['Unnamed: 0','extract','tokens','CODE'], axis=1, inplace = True)
#     X = X.values
#     y = feature_file_df.iloc[:, -1].to_frame()
#     y = y.values
#     #     df.iloc[:,-1]
#     skf = StratifiedKFold(n_splits=10)
#     #     print(skf)
#     skf.get_n_splits(X, y)
#     # remove_stop_words = False

from sklearn.model_selection import KFold
#initialize  k fold

kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(X.values)

accuracy_list = []
fold = 0

# cols = feature_file_df.columns
# print(cols)

from imblearn.over_sampling import SMOTE

for train_index, test_index in kf.split(X.values):
    print("TRAIN:", train_index, "TEST:", test_index)
    print("TRAIN Size:", len(train_index), "TEST Size:", len(test_index))

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train_df = X.iloc[train_index]
    y_train_df = y.iloc[train_index]
    X_test_df = X.iloc[test_index]
    y_test_df = y.iloc[test_index]

    X_train_reshape, y_train_reshape_matrix, X_test_reshape, y_test_reshape_matrix = reshape_data(X_train_df, X_test_df,
                                                                                                  y_train_df, y_test_df)

    val_lost, val_acc = simple_LSTM(X_train_reshape, y_train_reshape_matrix, X_test_reshape, y_test_reshape_matrix, vector_size,
                model_batch_size, no_epochs)

    #append all accuracies
    #may want to have a model and save each model separately
    #also include option logic to control stopwords
    accuracy_list.append(val_acc)

    # X_train = X_train_df.values
    # y_train = y_train_df.values
    # X_test = X_test_df.values
    # y_test = y_test_df.values

#     print("Number transactions X_train dataset: ", X_train.shape)
#     print("Number transactions y_train dataset: ", y_train.shape)
#     print("Number transactions X_test dataset: ", X_test.shape)
#     print("Number transactions y_test dataset: ", y_test.shape)
#
#     print("Before OverSampling, counts of label '6': {}".format(sum(y_train == 6)))
#     print("Before OverSampling, counts of label '5': {}".format(sum(y_train == 5)))
#     print("Before OverSampling, counts of label '4': {}".format(sum(y_train == 4)))
#     print("Before OverSampling, counts of label '3': {}".format(sum(y_train == 3)))
#     print("Before OverSampling, counts of label '2': {}".format(sum(y_train == 2)))
#     print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
#     print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))
#
#     print(X_train)
#     print(y_train.ravel())
#     sm = SMOTE(random_state=2)
#     X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
#
#     print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
#     print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
#
#     print("Before OverSampling, counts of label '6': {}".format(sum(y_train_res == 6)))
#     print("Before OverSampling, counts of label '5': {}".format(sum(y_train_res == 5)))
#     print("Before OverSampling, counts of label '4': {}".format(sum(y_train_res == 4)))
#     print("Before OverSampling, counts of label '3': {}".format(sum(y_train_res == 3)))
#     print("Before OverSampling, counts of label '2': {}".format(sum(y_train_res == 2)))
#     print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
#     print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))
#
#     # Classification intensity score histogram
#     #         plt.figure()
#     #         ax = sns.countplot(x = 'CODE', data=y_train_df);
#     #         ax.set_title('Code Classification Distribution')
#     #         ax.set_ylabel('Count')
#     #         handles, labels = ax.get_legend_handles_labels()
#
#     X_train_res_df = pd.DataFrame(X_train_res)
#     y_train_res_df = pd.DataFrame(y_train_res)
#     X_train_df = pd.DataFrame(X_train)
#     y_train_df = pd.DataFrame(y_train)
#     X_test_df = pd.DataFrame(X_test)
#     y_test_df = pd.DataFrame(y_test)
#
#     remove_stop_words = False
#     accuracy = rf_model(X_train_res_df, y_train_res_df, X_test_df, y_test_df, remove_stop_words, fold)
#     #     #append all accuracies
#     #     #may want to have a model and save each model separately
#     #     #also include option logic to control stopwords
#     accuracy_list.append(accuracy)
#
    fold = fold + 1
#
if (model_count == 0):
    results = pd.DataFrame(np.array(accuracy_list)).T

    # append index and text onto dataframe
    # results['embedding'] = row['paths']
    results['embedding'] = 'universal encoding'

else:
    results_2 = pd.DataFrame(np.array(accuracy_list)).T
    # results_2['embedding'] = row['paths']
    results_2['embedding'] = 'next_embedding'

    results = results.append(results_2)

print("results:")
print(results)
#
# print("model_count: " + str(model_count))
# model_count += 1