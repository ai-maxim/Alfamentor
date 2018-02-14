import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from definitions import ROOT_DIR, IMPORT_NAME, TAGS_NAME

class Data_Prep:
# Data loading
    def data_load(self):

        df_in = pd.read_csv(os.path.join(ROOT_DIR + "/" + IMPORT_NAME), index_col=0)
        with open(os.path.join(ROOT_DIR + "/" + TAGS_NAME)) as f:
            tags = f.readlines()

        return df_in, tags

    # Dataset treatment before word2vec indexation
    def pretreatment(self, df_in, tags):

        # Tags = columns labels
        tag_ls = tags[0].split(",")
        tag_ls.append('grant')
        # Copy needed colons to new datafame
        df_t = df_in[tag_ls].copy()
        tag_ls.remove('grant')

        # Cleaning Dataframe from symbols
        df_t = df_t.replace(['\n', '\r'], ['', ''], regex=True)
        df_t = df_t.replace('[]',float('nan'))

        for i in tag_ls:
            #print(i)
            df_t = df_t.loc[df_in[i].notnull()]
            for ind in df_t[i]:
                if (type(ind) == str):
                    ind.strip(' ')

        #Dividing Dataframe to Positive and Negative

        df_neg = df_t.loc[df_in['grant'] == 0]
        df_pos = df_t.loc[df_in['grant'] == 1]

        df_neg = df_neg[tag_ls]
        df_neg.reset_index()

        df_pos = df_pos[tag_ls]
        df_pos.reset_index()

        return df_pos, df_neg

    # Saving data
    def data_save(self,df_pos,df_neg):
        df_pos.to_csv(os.path.join(ROOT_DIR + "/DF_POS.csv"), encoding='utf-8')
        df_neg.to_csv(os.path.join(ROOT_DIR + "/DF_NEG.csv"), encoding='utf-8')
        return 0


# Main code
data = Data_Prep()
df_in, tags = data.data_load()
df_pos, df_neg = data.pretreatment(df_in, tags)
data.data_save(df_pos, df_neg)

