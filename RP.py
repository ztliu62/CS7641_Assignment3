import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from helpers import   pairwiseDistCorr,nn_reg,nn_arch,reconstructionError
from sklearn.random_projection import SparseRandomProjection
from itertools import product

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def barplot_abalone(tmp, num_range):
	ax1 = plt.figure().gca()
	ax1.plot(num_range, tmp, color='blueviolet', linestyle='-', marker='.')
	ax1.set_xlabel('# of Dimension')
	ax1.set_ylabel('Accuracy(%)')
	plt.title('Abalone RP Dimensionality Reduction')
	plt.grid()
	plt.savefig('./abalone_RP_analysis.png')

def barplot_breast(tmp, num_range):
	ax1 = plt.figure().gca()
	ax1.plot(num_range, tmp, color='blueviolet', linestyle='-', marker='.')
	ax1.set_xlabel('# of Dimension')
	ax1.set_ylabel('Accuracy(%)')
	plt.title('Breast RP Dimensionality Reduction')
	plt.grid()
	plt.savefig('./Breast_RP_analysis.png')

def calculate_accuracy(X, y, dims):

	X_trg, X_test, y_trg, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
	clf = DecisionTreeClassifier(random_state = 5)
	clf.fit(X_trg, y_trg)
	y_pred = clf.predict(X_test)
	accuracy_bm = accuracy_score(y_pred, y_test)

	ratio = []
	for i in dims:
		rp = SparseRandomProjection(random_state=5, n_components=i)   
		clf = DecisionTreeClassifier(random_state = 5)
		pipe = Pipeline([('rp', rp), ('clf', clf)])
		pipe.fit(X_trg,y_trg)
		y_pred = pipe.predict(X_test)
		accuracy = accuracy_score(y_pred, y_test)
		ratio.append(accuracy/accuracy_bm)

	return ratio 

# Breast Cancer Dataset
br = pd.read_csv('./BASE/breast.csv')
brX = br.drop('Class', 1).copy().values
brY = br['Class'].copy().values
brX = StandardScaler().fit_transform(brX)

cluster_range = range(1,11)
dims = range(1,30)

tmp = defaultdict(dict)
for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(brX), brX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv('./RP/breast_scree.csv')

ratio = calculate_accuracy(brX, brY, range(1,30))
barplot_breast(ratio, range(1,30))


dim = 10
rp = SparseRandomProjection(n_components=dim,random_state=5)

brX2 = rp.fit_transform(brX)
br2 = pd.DataFrame(np.hstack((brX2,np.atleast_2d(brY).T)))
cols = list(range(br2.shape[1]))
cols[-1] = 'Class'
br2.columns = cols
br2.to_csv('./RP/breast.csv')


# Abalone Dataset
abalone = pd.read_csv('./BASE/abalone.csv')
abaloneX = abalone.drop('Class', 1).copy().values
abaloneY = abalone['Class'].copy().values
abaloneX = StandardScaler().fit_transform(abaloneX)

cluster_range = range(1,11)
dims = range(1,10)

tmp = defaultdict(dict)
for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(abaloneX), abaloneX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv('./RP/abalone_scree.csv')

ratio = calculate_accuracy(abaloneX, abaloneY, range(1,10))
barplot_abalone(ratio, range(1,10))

dim = 6
rp = SparseRandomProjection(n_components=dim,random_state=5)

abaloneX2 = rp.fit_transform(abaloneX)
abalone2 = pd.DataFrame(np.hstack((abaloneX2,np.atleast_2d(abaloneY).T)))
cols = list(range(abalone2.shape[1]))
cols[-1] = 'Class'
abalone2.columns = cols
abalone2.to_csv('./RP/abalone.csv')
