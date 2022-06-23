import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.ensemble import IsolationForest

df = pd.read_csv("dataset/Arrhythmia.csv")
df['class'].describe()

plt.scatter(range(df.shape[0]), np.sort(df['class'].values))
plt.xlabel('index')
plt.ylabel('Class')
plt.title("Arrhythmia Distribution")
sns.despine()

sns.distplot(df['class'])
plt.title("Distribution of Arrhythmia")
sns.despine()
