# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 00:30:36 2022

@author: USUARIO
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

import pickle


df = pd.read_csv('https://raw.githubusercontent.com/carrionalfredo/Spaceship-Titanic/main/train.csv')


##Data preparation adn cleaning
df.columns = df.columns.str.lower().str.replace(' ' , '_')

##Convert 'cryosleep' and 'vip' columns from object to boolean
df.cryosleep = df.cryosleep.astype(bool)
df.vip = df.vip.astype(bool)

##split the 'cabin' columns into 3 new columns: 'deck', 'num' and 'side'
cabin = df['cabin'].str.split('/', expand = True)
cabin.columns = ['deck', 'num', 'side']
cabin.num = pd.to_numeric(cabin.num)
 ##joint the 3 new columns to dataframe and delete the 'cabin' column
df = pd.concat([df, cabin], axis=1)
del df['cabin']

##identify types of variables
categorical = list(df.dtypes[df.dtypes == 'object'].index)
boolean = list(df.dtypes[df.dtypes == 'bool'].index)
numerical = list(df.dtypes[df.dtypes == 'float64'].index)

##data cleaning and fill na's
df[categorical] = df[categorical].fillna('unk')
for c in categorical:
    df[c] = df[c].str.lower().str.replace(' ' , '_')

#for the 'age' columns, the na's replacement will be with the mean of 'age'
df['age'] = df['age'].fillna(df['age'].mean())

##for numerical variables, the na's replacement will be with zeros
df[numerical] =df[numerical].fillna(0)

##selection of variables for training the model
##the 'passengerid' and 'name' columns are descarted from the dataframe
categorical_columns = ['homeplanet', 'destination', 'deck', 'side', 'cryosleep', 'vip', 'transported']

##preparation of the data to train the model (60/20/20 ratio)
df_fulltrain, df_test = train_test_split(
    df[categorical_columns+numerical],
    test_size=0.2,
    random_state=1
    )
df_train, df_val = train_test_split(
    df_fulltrain[categorical_columns+numerical],
    test_size=0.25,
    random_state=1
    )

df_train = df_train.reset_index(drop = True)
y_train = df_train.transported.values

##delete the objective variable 'transported'
del df_train['transported']

##one hot encoding fo the selected variables (29 variables)
train_dicts= df_train.to_dict(orient='records')
dv = DictVectorizer(sparse= False)
dv.fit(train_dicts)
x_train = dv.fit_transform(train_dicts)

##model training
model=LogisticRegression(max_iter=2000, C=6, class_weight='balanced')
model.fit(x_train, y_train)


##saving the model
with open('LGRmodel.bin', 'wb') as f_out:
    pickle.dump((dv,model), f_out)