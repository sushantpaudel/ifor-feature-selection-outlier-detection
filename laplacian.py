from scipy.io import arff
#from skfeature.function.similarity_based import lap_score
#from skfeature.utility import construct_W
from LaplacianScoreMethod import construct_W
from LaplacianScoreMethod import lap_score

mat = arff.loadarff('./dataset/Annthyroid.arff')
df = pd.DataFrame(data[0])
df.pop('Target')

# construct affinity matrix
kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
W = construct_W.construct_W(X, **kwargs_W)

# obtain the scores of features
score = lap_score.lap_score(X, W=W)

# sort the feature scores in an ascending order according to the feature scores
idx = lap_score.feature_ranking(score)

print('Score:', score)
print('Index:', idx)