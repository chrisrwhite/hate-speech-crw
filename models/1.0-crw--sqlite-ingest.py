##################################################
## Uploads CSV data to sqlite database
## hate speech
## toxic
##################################################
## Author: Christopher White
##################################################

import csv, sqlite3
import pandas as pd
db_path = '../../data/processed/test.db'


# dataset = 'hate'
dataset = 'toxic'


con = sqlite3.connect(db_path)
cur = con.cursor()



if dataset == 'hate':
    raw_path = '../../data/raw/consolidated_data_4_10_2019.csv'
    cur.execute("DROP TABLE IF EXISTS hate")
    cur.execute("DROP TABLE IF EXISTS t")

    columns = ['show_date', 'show_name', 'subject', 'keyword', 'extract','A_B', 'CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6']
    print(['%s text' % column for column in columns])
    cur.execute("CREATE TABLE hate ({} datetime,{} str,{} str,{} str,{} str,{} int,{} str,{} str,{} str,{} str,{} str,{} str,{} str);".format('show_date','show_name','subject','keyword','extract','A_B', 'CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6')) # use your column names here
    raw_data_df = pd.read_csv(raw_path, encoding='latin-1')

    print(raw_data_df.head())

    y_label = raw_data_df['CODE']
    y_train_reshape = pd.get_dummies(y_label.astype(str))

    print(y_train_reshape.head())

    y_train_reshape.columns = ['CODE_0', 'CODE_1', 'CODE_2', 'CODE_3', 'CODE_4', 'CODE_5', 'CODE_6']

    print(y_train_reshape.head())

    raw_data_df=raw_data_df.join(y_train_reshape)

    raw_data_df.drop(['CODE'], axis=1, inplace=True)
    print(raw_data_df.head())

    raw_data_df.to_sql('hate', con, if_exists='append', index = False) # Insert the values from the csv file into the table 'CLIENTS'

    # output_name = 'hate_clean_nostop'

else:
    raw_path = '../../data/raw/jigsaw-toxic-comment-classification-challenge/train.csv'
    cur.execute("DROP TABLE IF EXISTS toxic_temp")
    cur.execute("DROP TABLE IF EXISTS toxic")
    columns = ["id", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    print(['%s text' % column for column in columns])
    cur.execute(
        "CREATE TABLE toxic_temp ({} str,{} str,{} str,{} str,{} str,{} str,{} str,{} str);".format("id", "comment_text",
                                                                                               "toxic", "severe_toxic",
                                                                                               "obscene", "threat",
                                                                                               "insult",
                                                                                               "identity_hate"))  # use your column names here
    raw_data_df = pd.read_csv(raw_path, encoding='latin-1')

    raw_data_df.to_sql('toxic_temp', con, if_exists='append',
                       index=False)  # Insert the values from the csv file into the table 'CLIENTS'

    cur.execute("CREATE TABLE toxic AS SELECT id, comment_text AS extract, toxic, severe_toxic, obscene, threat, insult, identity_hate  FROM toxic_temp")
    cur.execute("DROP TABLE IF EXISTS toxic_temp")


    # output_name = 'toxic_clean_nostop'





