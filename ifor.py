import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from random import randrange

#########################################################
# Getting the dataset
df = pd.read_csv("./dataset/Arrhythmia.csv")
# Dropping the data with class as, this is an unsupervised learning
df.drop('class', inplace=True, axis=1)
#  contamination for training in IFOR
contamination = 0.01

df.dropna(inplace=True)
data = df.copy()


model = IsolationForest(contamination=contamination,n_estimators=1000)
model.fit(data)

df['iforest'] = pd.Series(model.predict(data))
df['iforest'] = df['iforest'].map({1:0,-1:1})
print("----------------------------")
print(df['iforest'].value_counts())
