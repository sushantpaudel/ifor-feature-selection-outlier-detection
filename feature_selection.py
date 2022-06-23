
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import sys

def chisquare(noOfFeatures = 20):
  data = pd.read_csv("./dataset/Arrhythmia.csv")
  X = data.iloc[:,0:-1]  #independent variable columns
  y = data.iloc[:,-1]    #target variable column (price range)
  
  print(np.isnan(X.any())) #and gets False
  print(np.isfinite(X.all())) #and gets True



  #extracting top 10 best features by applying SelectKBest class
  bestfeatures = SelectKBest(score_func=chi2, k=10)
  fit = bestfeatures.fit(X,y)
  dfscores = pd.DataFrame(fit.scores_)
  dfcolumns = pd.DataFrame(X.columns)
  
  #concat two dataframes
  featureScores = pd.concat([dfcolumns,dfscores],axis=1)
  featureScores.columns = ['Specs','Score']  #naming the dataframe columns
  print(featureScores.nlargest(noOfFeatures,'Score'))  #printing 10 best features



arguments = sys.argv
"""
  eg: python feature_selection.py chi2
  1. chi2 -> Chi square feature selection method
"""
try:
  typeOfAlgorithm = arguments[1]
except:
  typeOfAlgorithm = None

if typeOfAlgorithm == "chi2":
  chisquare()
else:
  print("You have to provide a feature selection method, eg: $ python feature_selection.py chi2")