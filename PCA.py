import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from helpers import nn_arch,nn_reg

np.random.seed(42)

def barplot(tmp, cumulative, dataset):
	fig, ax1 = plt.subplots(figsize = (8,5))
	ax2 = ax1.twinx()
	ax1.bar(range(0,10), tmp[0:10], color='olivedrab', label='Eigenvalue')
	ax1.legend(loc = 'center right')
	ax1.set_xlabel('# of Principle Components')
	ax1.set_ylabel('EigenValue')

	ax2.plot(range(0, 10), cumulative[0:10], color = 'blueviolet', linestyle='-', marker='.')
	ax2.set_ylabel('Cumulative Ratio(%)')
	ax2.set_ylim([0, 1.])
	ax2.legend(loc = 'best')

	plt.title(dataset + 'PCA Dimensionality Reduction')
	plt.grid()
	plt.savefig('./' + dataset + '_PCA_analysis.png')

# Breast Cancer Dataset
br = pd.read_csv('./BASE/breast.csv')
brX = br.drop('Class', 1).copy().values
brY = br['Class'].copy().values
brX = StandardScaler().fit_transform(brX)

cluster_range = range(1,11)

pca = PCA(random_state = 5)
pca.fit(brX)
tmp = pd.Series(data = pca.explained_variance_,index = range(0,30))
tmp.to_csv('./PCA/breast scree.csv')
tmp2 = pd.Series(data = pca.explained_variance_ratio_,index = range(0,30))
cumulative = []
cur = 0 
for i in range(0,30):
	cur += tmp2[i]
	cumulative.append(cur)

barplot(tmp, cumulative, 'Breast_Cancer')

dim = 10
pca = PCA(n_components = dim, random_state = 5)
brX2 = pca.fit_transform(brX)
br2 = pd.DataFrame(np.hstack((brX2,np.atleast_2d(brY).T)))
cols = list(range(br2.shape[1]))
cols[-1] = 'Class'
br2.columns = cols
br2.to_csv('./PCA/breast.csv')



# Abalone Dataset
abalone = pd.read_csv('./BASE/abalone.csv')
abaloneX = abalone.drop('Class', 1).copy().values
abaloneY = abalone['Class'].copy().values
abaloneX = StandardScaler().fit_transform(abaloneX)

cluster_range = range(1,10)

pca = PCA(random_state = 5)
pca.fit(abaloneX)
tmp = pd.Series(data = pca.explained_variance_,index = range(0,10))
tmp.to_csv('./PCA/abalone_scree.csv')
tmp2 = pd.Series(data = pca.explained_variance_ratio_,index = range(0,10))
cumulative = []
cur = 0 
for i in range(0,10):
	cur += tmp2[i]
	cumulative.append(cur)

barplot(tmp, cumulative, 'Abalone')

dim = 6
pca = PCA(n_components = dim, random_state = 5)
abaloneX2 = pca.fit_transform(abaloneX)
abalone2 = pd.DataFrame(np.hstack((abaloneX2,np.atleast_2d(abaloneY).T)))
cols = list(range(abalone2.shape[1]))
cols[-1] = 'Class'
abalone2.columns = cols
abalone2.to_csv('./PCA/abalone.csv')
