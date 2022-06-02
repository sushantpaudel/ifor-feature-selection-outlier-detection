from scipy.io import arff
import pandas as pd

data = arff.loadarff('dataset/Annthyroid.arff')
df = pd.DataFrame(data[0])

df.pop('Target')

print(df.head())
