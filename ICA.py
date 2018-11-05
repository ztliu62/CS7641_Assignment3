import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA

def barplot_breast(tmp):
	ax1 = plt.figure().gca()
	ax1.bar(range(1,30), tmp, color='olivedrab', label='Kurtosis')
	ax1.legend(loc = 'center right')
	ax1.set_xlabel('# of Independent Components')
	ax1.set_ylabel('Kurtosis')

	plt.title('Breast-Cancer ICA Dimensionality Reduction')
	plt.grid()
	plt.savefig('./breast_ICA_analysis.png')

def barplot_abalone(tmp):
	ax1 = plt.figure().gca()
	ax1.bar(range(1,10), tmp, color='olivedrab', label='Kurtosis')
	ax1.legend(loc = 'center right')
	ax1.set_xlabel('# of Independent Components')
	ax1.set_ylabel('Kurtosis')

	plt.title('Abalone ICA Dimensionality Reduction')
	plt.grid()
	plt.savefig('./abalone_ICA_analysis.png')

# Breast Cancer Dataset
br = pd.read_csv('./BASE/breast.csv')
brX = br.drop('Class', 1).copy().values
brY = br['Class'].copy().values
brX = StandardScaler().fit_transform(brX)

cluster_range = range(1,11)
dims = range(1,30)

ica = FastICA(random_state=42, max_iter = 1000)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(brX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt) 
kurt.to_csv('./ICA/breast_scree.csv')
barplot_breast(np.sort(kurt)[::-1])

dim = 10
ica = FastICA(n_components=dim,random_state=5) # try to add more iterations
brX2 = ica.fit_transform(brX)
br2 = pd.DataFrame(np.hstack((brX2,np.atleast_2d(brY).T)))
cols = list(range(br2.shape[1]))
cols[-1] = 'Class'
br2.columns = cols
br2.to_csv('./ICA/breast.csv')


# Abalone Dataset
abalone = pd.read_csv('./BASE/abalone.csv')
abaloneX = abalone.drop('Class', 1).copy().values
abaloneY = abalone['Class'].copy().values
abaloneX = StandardScaler().fit_transform(abaloneX)

cluster_range = range(1,11)
dims = range(1,10)

ica = FastICA(random_state=42)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(abaloneX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt) 
kurt.to_csv('./ICA/abalone_scree.csv')
barplot_abalone(np.sort(kurt)[::-1])

dim = 6
ica = FastICA(n_components=dim,random_state=5) # try to add more iterations
abaloneX2 = ica.fit_transform(abaloneX)
abalone2 = pd.DataFrame(np.hstack((abaloneX2,np.atleast_2d(abaloneY).T)))
cols = list(range(abalone2.shape[1]))
cols[-1] = 'Class'
abalone2.columns = cols
abalone2.to_csv('./ICA/abalone.csv')