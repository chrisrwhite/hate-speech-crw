import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
#need to update input file to ESN Clean _Nostop, on my desktop

# path_input = "../data/processed/sentence_embeddings/infersent_feature_file_without_stopwords_clean.csv"

# path_input = "../data/processed/sentence_embeddings/tox_infersent_feature_file_without_stopwords_clean.csv"

path_input = "../data/processed/sentence_embeddings/universal_feature_file_without_stopwords.csv"



db_path = '../data/processed/test.db'
# db = sqlite3.connect(db_path)
con = sqlite3.connect(db_path)
cur = con.cursor()



dataset = 'hate_nostop'
# dataset = 'toxic_nostop'

# def missing_values_table(df):
#     mis_val = df.isnull().sum()
#     mis_val_percent = 100 * df.isnull().sum() / len(df)
#     mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
#     mis_val_table_ren_columns = mis_val_table.rename(
#         columns={0: 'Missing Values', 1: '% of Total Values'})
#     mis_val_table_ren_columns = mis_val_table_ren_columns[
#         mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
#         '% of Total Values', ascending=False).round(1)
#     print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
#                                                               "There are " + str(mis_val_table_ren_columns.shape[0]) +
#           " columns that have missing values.")
#     return mis_val_table_ren_columns


if dataset == 'hate_nostop':
    X = pd.read_sql_query("SELECT * from hate_universal_encoder_embedding_wefat",con)

    # fill nan wefat scores
    # missing_values_table(X)
    X = X.fillna(0)
    # missing_values_table(X)
    cols = X.columns
    print(cols)

    print('hate full feature file')
    print(X['extract'].head())
    print(X.shape)
    # print(X.dtypes)



    # separate features from label
    # y_cols = ['CODE', 'CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6']
    y_cols = ['CODE']

    y = X.loc[:, y_cols]
    # X.drop(['id','extract','CODE','CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6'], axis=1, inplace = True)
    X.drop(['index','id', 'extract', 'CODE'], axis=1,
           inplace=True)





    # need to make sure i am separating a test dataset completely from the neural network training
    # https://stackoverflow.com/questions/46308374/what-is-validation-data-used-for-in-a-keras-sequential-model

    # random_seed = 42 # acc = 0.23
    # random_seed = 10 # acc = 0.28
    # random_seed = 15 # acc = 0.28
    # random_seed = 101 # 0.219
    # random_seed = 1 #0.22
    # random_seed = 11 # 0.30
    # random_seed = 111 # 0.309
    # random_seed = 1010 # 0.35
    # random_seed = 10101  #0.218
    random_seed = 123 # 0.41
    # random_seed = 132  # 0.279
    # random_seed = 321 # 0.253
    # random_seed = 320 #0.227
    # random_seed = 404 #0.26
    # random_seed = 202 # 0.27
    # random_seed = 808


    X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    print('X_training')
    print(X_training.head())
    print('y_training')
    print(y_training.head())

    # output_name = 'hate_clean_nostop'

else:

    X_training = pd.read_sql_query("SELECT * from toxic_universal_encoder_embedding_features", con)

    print(X_training.head())
    print(X_training.dtypes)


    y_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y_training = X_training.loc[:, y_cols]
    X_training.drop(['extract', "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
           axis=1, inplace=True)

    print(X_training.head())
    print(y_training.head())


    # output_name = 'toxic_clean_nostop'


print('creating train and test sets')

import numpy as np
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# X_train_full_trial_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
# X_test_full_trial_df = pd.merge(X_test, y_test, left_index=True, right_index=True)
#
# y_train = X_train_full_trial_df[['CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6']]
# X_train = X_train_full_trial_df.drop(
#     ['CODE', 'CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6'], axis=1)
#
# y_test = X_test_full_trial_df[['CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6']]
# X_test = X_test_full_trial_df.drop(['CODE', 'CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6'], axis=1)
#
# X_train_np = np.array(X_train)
# X_test_np = np.array(X_test)
# y_train_np = np.array(y_train)
# y_test_np = np.array(y_test)
#
# print("Number transactions X_train dataset: ", X_train.shape)
# print("Number transactions y_train dataset: ", y_train.shape)
# print("Number transactions X_test dataset: ", X_test.shape)
# print("Number transactions y_test dataset: ", y_test.shape)


# convert numpy stuff back into pandas dataframes
# y_cols_new = ['CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6']
# print("y_cols_new: " + str(y_cols_new))
#
# X_train = pd.DataFrame(X_train)
# y_train = pd.DataFrame(y_train, columns = y_cols_new)
#
#
# X_test = pd.DataFrame(X_test_np)
# y_test = pd.DataFrame(y_test_np, columns = y_cols_new)
#
# print(y_test.head())



# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
# The three dimensions of LSTM input are:

# 1. Samples. One sequence is one sample. A batch is comprised of one or more samples.
# 2. Time Steps. One time step is one point of observation in the sample.
# 3. Features. One feature is one observation at a time step.

# batch_size = 1

import numpy as np



def reshape_data_model_1(X_train, X_val, y_train, y_val):
    X_train_reshape = np.reshape(X_train.values, ( X_train.shape[0],1, X_train.shape[1]))

    X_val_reshape = np.reshape(X_val.values, ( X_val.shape[0],1, X_val.shape[1]))

    y_train_reshape_matrix = y_train.values
    y_train_reshape_matrix = y_train_reshape_matrix.reshape(y_train.shape[0],1,y_train.shape[1])

    y_val_reshape_matrix = y_val.values
    y_val_reshape_matrix = y_val_reshape_matrix.reshape(y_val.shape[0],1,y_val.shape[1])


    print(X_train_reshape.shape)

    print(y_train_reshape_matrix.shape)

    print(X_val_reshape.shape)

    print(y_val_reshape_matrix.shape)

    return X_train_reshape, y_train_reshape_matrix,X_val_reshape, y_val_reshape_matrix

def reshape_data_model_2(X_train, X_val, y_train, y_val):
    X_train_reshape = np.reshape(X_train.values, ( X_train.shape[0], X_train.shape[1], 1))

    X_val_reshape = np.reshape(X_val.values, ( X_val.shape[0], X_val.shape[1],1))

    y_train_reshape_matrix = y_train.values
    y_train_reshape_matrix = y_train_reshape_matrix.reshape(y_train.shape[0],y_train.shape[1],1)

    y_val_reshape_matrix = y_val.values
    y_val_reshape_matrix = y_val_reshape_matrix.reshape(y_val.shape[0],y_val.shape[1],1)


    print(X_train_reshape.shape)

    print(y_train_reshape_matrix.shape)

    print(X_val_reshape.shape)

    print(y_val_reshape_matrix.shape)

    return X_train_reshape, y_train_reshape_matrix,X_val_reshape, y_val_reshape_matrix

# X_train_reshape, y_train_reshape_matrix,X_val_reshape, y_val_reshape_matrix = reshape_data(X_train, X_val, y_train, y_val)

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
from keras.models import load_model

#LSTM
def simple_LSTM(X_train_reshape, y_train_reshape_matrix,X_val_reshape, y_val_reshape_matrix,vector_size,model_batch_size,no_epochs):






    # We usually match up the size of the embedding layer output with the number of hidden layers in the LSTM cell.
    # https://adventuresinmachinelearning.com/keras-lstm-tutorial/
    # hidden_size = 4096
    # hidden_size = 32
    hidden_size = 512

    from keras.models import Model
    from keras.layers import Input, Dense
    from keras.layers import TimeDistributed
    from keras.layers import Conv1D, Dropout, Dense, Flatten, LSTM, MaxPooling1D, Bidirectional
    from keras.optimizers import Adam
    from keras.layers import Flatten
    DROP_RATE_DENSE = 0.1



    #perahaps i want to use a combinatinos of convolutional and lstm?
    # https://github.com/jatinmandav/Neural-Networks/blob/master/Sentiment-Analysis/universal-sentence-encoder/universal_sentence_encoder_sentiment-analysis.ipynb


    ####################### MODEL 1: ~ 30% accuracy ####################################


    # use variable length timesteps using None
    # https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras

    # model = Sequential()
    # timesteps = None
    # data_dim = X_train_reshape.shape[2]
    #
    # model = Sequential()
    # input = Input(shape=(timesteps, data_dim))
    # model = Bidirectional(LSTM(hidden_size, return_sequences=True, input_shape=(timesteps, data_dim)),merge_mode='concat')(input) #accuracy  = 0.316
    # model = Bidirectional(LSTM(hidden_size, return_sequences=True), merge_mode='concat')(input)  # accuracy  = 0.279
    # # model = Bidirectional(LSTM(hidden_size, return_sequences=True, input_shape=(timesteps, data_dim)),merge_mode='concat')(model) # 3) added this 0.21259
    # # model = Bidirectional(LSTM(hidden_size, return_sequences=True, input_shape=(timesteps, data_dim)), merge_mode='concat')(model)
    # model = Dropout(DROP_RATE_DENSE)(model)
    #
    # model = TimeDistributed(Dense(256, activation='relu'))(model) #accuracy after adding this line  =0.318
    # # model = Flatten(input_shape=(timesteps, data_dim))(model)
    # # model = Dense(100, activation='relu')(model)
    # model = Dropout(DROP_RATE_DENSE)(model)
    #
    # output = Dense(y_train_reshape_matrix.shape[2], activation='softmax')(model)
    # model = Model(input, output)
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

    ####################### MODEL 1:                 ####################################

    ####################### MODEL 2: ~ 30% accuracy ####################################

    # use variable length timesteps using None
    # https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras

    model = Sequential()
    timesteps = None
    data_dim = X_train_reshape.shape[2]
    vector_size = 512


    model = Sequential()
    input = Input(shape=(timesteps, data_dim))
    model = Conv1D(32, kernel_size=3, activation='elu', padding='same', input_shape=(vector_size, 1))(input)
    model = Conv1D(32, kernel_size=3, activation='elu', padding='same')(model)
    model = Conv1D(32, kernel_size=3, activation='relu', padding='same')(model)
    model = MaxPooling1D(pool_size=3)(model)
    model = Bidirectional(LSTM(hidden_size, return_sequences=True, input_shape=(timesteps, data_dim)),
                          merge_mode='concat')(model)  # accuracy  = 0.316
    model = Bidirectional(LSTM(hidden_size, return_sequences=True), merge_mode='concat')(input)  # accuracy  = 0.279
    model = Bidirectional(LSTM(hidden_size, return_sequences=True),merge_mode='concat')(model) # 3) added this 0.21259
    model = Bidirectional(LSTM(hidden_size, return_sequences=True), merge_mode='concat')(model)


    output = Dense(y_train_reshape_matrix.shape[2], activation='softmax')(model)
    model = Model(input, output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

    ####################### MODEL 2:                 ####################################


















    # from keras.models import Sequential
    # from keras.layers import Conv1D, Dropout, Dense, Flatten, LSTM, MaxPooling1D, Bidirectional
    # from keras.optimizers import Adam
    # from keras.callbacks import EarlyStopping, TensorBoard
    #
    # model = Sequential()
    #
    # vector_size = 512
    # model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same',
    #                  input_shape=(vector_size,1)))
    # model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
    # model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    # model.add(MaxPooling1D(pool_size=3))
    #
    # model.add(Bidirectional(LSTM(512, dropout=0.2, recurrent_dropout=0.3)))
    #
    # model.add(Dense(512, activation='sigmoid'))
    # model.add(Dropout(0.2))
    # model.add(Dense(512, activation='sigmoid'))
    # model.add(Dropout(0.25))
    # model.add(Dense(512, activation='sigmoid'))
    # model.add(Dropout(0.25))
    #
    #
    # model.add(Dense(y_train_reshape_matrix.shape[1], activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])


    print(model.summary())
    # Keras detects the output_shape and automatically determines which accuracy to use when accuracy is specified. For multi-class classification, categorical_accuracy will be used internally.
    # https://stackoverflow.com/questions/43544358/categorical-crossentropy-need-to-use-categorical-accuracy-or-accuracy-as-the-met
    tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, write_graph=True, write_images=True)


    # print('data_dim shape: ' + str(data_dim))
    print('X_train_reshape shape: ' + str(X_train_reshape.shape))
    print('y_train_reshape_matrix shape: ' + str(y_train_reshape_matrix.shape))


    # model.fit(X_train_reshape,y_train_reshape_matrix, batch_size=500, epochs=15)
    model.fit(X_train_reshape, y_train_reshape_matrix, batch_size=model_batch_size, epochs=no_epochs,
             validation_data=(X_val_reshape, y_val_reshape_matrix), callbacks=[tensorboard, EarlyStopping(min_delta=0.0001, patience=3)])

    # prevent overfitting or over training of the network, EarlyStopping() is used in callback



    print('generate final prediction accuracy metrics')
    val_lost, val_acc = model.evaluate(X_val_reshape,y_val_reshape_matrix)

    print('final val_lost ' + str(val_lost))
    print('final val_acc ' + str(val_acc))

    return val_lost, val_acc, model


vector_size = 512
model_batch_size = 500
no_epochs = 25

# run LSTM
# val_lost, val_acc = simple_LSTM(X_train_reshape, y_train_reshape_matrix,X_val_reshape, y_val_reshape_matrix,vector_size,model_batch_size,no_epochs)

# result list to store accuracies across all folds ...  probably should output
# consolidated dataframe of all processed records

model_count = 0
# for index, row in df_files.iterrows():
#
#     feature_file_df = pd.read_csv(row['paths'])
#     #     valing feature files
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
from imblearn.over_sampling import SMOTE

#initialize  k fold

kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(X_training.values)

accuracy_list = []
fold = 0

# cols = feature_file_df.columns
# print(cols)


max_accuracy = 0

for train_index, val_index in kf.split(X_training.values):
    # print("TRAIN:", train_index, "val:", val_index)
    # print("TRAIN Size:", len(train_index), "val Size:", len(val_index))

    X_train, X_val = X_training.iloc[train_index], X_training.iloc[val_index]
    y_train, y_val = y_training.iloc[train_index], y_training.iloc[val_index]

    # create val and training sets based on the cross validation indexes
    X_train_df = X_training.iloc[train_index]
    y_train_df = y_training.iloc[train_index]
    X_val_df = X_training.iloc[val_index]
    y_val_df = y_training.iloc[val_index]

    if dataset == 'hate_nostop':
         #grab original X df column headers in  order to maintain CODE column names after numpy conversion

        X_cols = list(X_train_df.columns.values) + ['CODE']
        print(X_cols)
        print(len(X_cols))
        #SMOTE#################
        #convert to numpy
        X_train = X_train_df.values
        y_train = y_train_df.values
        X_val = X_val_df.values
        y_val = y_val_df.values

        # y_train_CODE = y_train_df['CODE'].values

        print("Number transactions X_train dataset: ", X_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_val dataset: ", X_val.shape)
        print("Number transactions y_val dataset: ", y_val.shape)

        print("Before OverSampling, counts of label '6': {}".format(sum(y_train == 6)))
        print("Before OverSampling, counts of label '5': {}".format(sum(y_train == 5)))
        print("Before OverSampling, counts of label '4': {}".format(sum(y_train == 4)))
        print("Before OverSampling, counts of label '3': {}".format(sum(y_train == 3)))
        print("Before OverSampling, counts of label '2': {}".format(sum(y_train == 2)))
        print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
        print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))


        # print(X_train)
        # print(y_train.ravel())
        # sm = SMOTE(random_state=2)
        sm = SMOTE()
        X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

        print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
        print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

        print("After OverSampling, counts of label '6': {}".format(sum(y_train_res == 6)))
        print("After OverSampling, counts of label '5': {}".format(sum(y_train_res == 5)))
        print("After OverSampling, counts of label '4': {}".format(sum(y_train_res == 4)))
        print("After OverSampling, counts of label '3': {}".format(sum(y_train_res == 3)))
        print("After OverSampling, counts of label '2': {}".format(sum(y_train_res == 2)))
        print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
        print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

        X_train_res_df = pd.DataFrame(X_train_res)

        y_train_res_df = pd.DataFrame(y_train_res)
        # X_train_df = pd.DataFrame(X_train)
        # y_train_df = pd.DataFrame(y_train)
        X_val_df = pd.DataFrame(X_val)

        y_val_df = pd.DataFrame(y_val)

        # Join X and y train and resplit with correct columns
        print(X_train_res_df.head())
        print(y_train_res_df.head())

        X_train_res_full_df = pd.merge(X_train_res_df, y_train_res_df, left_index=True, right_index=True)
        X_train_res_full_df.columns = X_cols
        X_val_full_df = pd.merge(X_val_df, y_val_df, left_index=True, right_index=True)
        X_val_full_df.columns = X_cols

        print(X_train_res_full_df.head())
        print(X_val_full_df.head())

        # X_train_res_full_df = X_train_res_df.join(y_train_res_df)
        # X_val_full_df = X_val_df.join(y_val_df)

        y_train_res_df = X_train_res_full_df[['CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6']]
        X_train_res_df = X_train_res_full_df.drop(['CODE', 'CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6'], axis=1)

        y_val_df = X_val_full_df[['CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6']]
        X_val_df = X_val_full_df.drop(['CODE', 'CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6'], axis=1)


        # SMOTE#################
        all_x_cols = X_train_res_df.columns
         # segment out word embedding and wefat scores
        wefat_cols = []
        embed_cols = []
        for col in all_x_cols:

             if col[-5:] == 'wefat':
                 # print(col)
                 # embed_cols.remove(col)
                 wefat_cols.append(col)
             else:
                 embed_cols.append(col)

        print('embed cols: ' + str(embed_cols))
        print('embed cols len: ' + str(len(embed_cols)))

        print('wefat cols: ' + str(wefat_cols))
        print('wefat cols len: ' + str(len(wefat_cols)))

        X_train_res_df = X_train_res_df[embed_cols]
        X_val_df = X_val_df[embed_cols]



        print('X and y dataframes')
        print(X_train_res_df.head())
        print(y_train_res_df.head())
        print(X_val_df.head())
        print(y_val_df.head())



         # modle 1
        X_train_reshape, y_train_reshape_matrix, X_val_reshape, y_val_reshape_matrix = reshape_data_model_1(X_train_res_df, X_val_df,
                                                                                                      y_train_res_df, y_val_df)





        val_lost, val_acc, model = simple_LSTM(X_train_reshape, y_train_reshape_matrix, X_val_reshape, y_val_reshape_matrix, vector_size,
                    model_batch_size, no_epochs)

        #model 2
        # X_train_reshape, y_train_reshape_matrix, X_val_reshape, y_val_reshape_matrix = reshape_data_model_2(
        #      X_train_res_df, X_val_df,
        #      y_train_res_df, y_val_df)
        # val_lost, val_acc, model = simple_LSTM(X_train_reshape, y_train_reshape_matrix, X_val_reshape, y_val_reshape_matrix,
        #                                vector_size,
        #                                model_batch_size, no_epochs)




    #toxic
    else:
        print('no toxic testing')
        # X_train_reshape, y_train_reshape_matrix, X_val_reshape, y_val_reshape_matrix = reshape_data(X_train_df, X_val_df,
        #                                                                                           y_train_df, y_val_df)
        #
        # val_lost, val_acc = simple_LSTM(X_train_reshape, y_train_reshape_matrix, X_val_reshape, y_val_reshape_matrix, vector_size,
        #         model_batch_size, no_epochs)

    break

    # save best performing model on validation dataset
    if val_acc > max_accuracy:
        # serialize weights to HDF5
        model.save("../models/model.h5")
        print("Saved model to disk")


    #append all accuracies
    #may want to have a model and save each model separately
    #also include option logic to control stopwords

    accuracy_list.append(val_acc)



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
#     X_val_df = pd.DataFrame(X_val)
#     y_val_df = pd.DataFrame(y_val)
#
#     remove_stop_words = False
#     accuracy = rf_model(X_train_res_df, y_train_res_df, X_val_df, y_val_df, remove_stop_words, fold)
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



# make a prediction


X_testing_full_df = pd.merge(X_testing, y_testing, left_index=True, right_index=True)
X_testing_full_df.columns = X_cols

print(X_testing_full_df.head())
# print(X_val_full_df.head())

# X_train_res_full_df = X_train_res_df.join(y_train_res_df)
# X_val_full_df = X_val_df.join(y_val_df)

y_testing_df = X_testing_full_df[['CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6']]
X_testing_df = X_testing_full_df[embed_cols]

X_testing_reshape = np.reshape(X_testing_df.values, ( X_testing_df.shape[0],1, X_testing_df.shape[1]))


#load best performing validation model:
model = load_model("../models/model.h5")

#model 2
# y_pred = model.predict_classes(X_testing_reshape)

#model 1
y_prob = model.predict(X_testing_reshape)
y_pred = y_prob.argmax(axis=-1)



# show the inputs and predicted outputs3

# print(ynew)
for i in range(len(X_testing)):
# 	# print("X=%s, Predicted=%s" % (X_testing.values[i], ynew[i]))
    print(y_pred[i])

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(X_testing_full_df['CODE'].values, y_pred)
print(cm)

print('accuracy on test set')

print(accuracy_score(X_testing_full_df['CODE'].values, y_pred))
