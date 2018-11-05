import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from helpers import ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def barplot_breast(tmp):
	ax1 = plt.figure().gca()
	ax1.bar(range(0,30), tmp, color='olivedrab', label='Importance')
	ax1.legend(loc = 'center right')
	ax1.set_xlabel('# of Features')
	ax1.set_ylabel('Importance')

	plt.title('Breast-Cancer RF Dimensionality Reduction')
	plt.grid()
	plt.savefig('./breast_RF_analysis.png')
	

def barplot_abalone(tmp):
	ax1 = plt.figure().gca()
	ax1.bar(range(0,10), tmp, color='olivedrab', label='Importance')
	ax1.legend(loc = 'center right')
	ax1.set_xlabel('# of Features')
	ax1.set_ylabel('Importance')

	plt.title('Abalone RF Dimensionality Reduction')
	plt.grid()
	plt.savefig('./abalone_RF_analysis.png')


# Breast Cancer Dataset
br = pd.read_csv('./BASE/breast.csv')
brX = br.drop('Class', 1).copy().values
brY = br['Class'].copy().values
brX = StandardScaler().fit_transform(brX)

cluster_range = range(1,11)
dims = range(1,30)

rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=-1)
fs_br = rfc.fit(brX,brY).feature_importances_ 
tmp = pd.Series(np.sort(fs_br)[::-1])
tmp.to_csv('./RF/breast_scree.csv')

barplot_breast(tmp)


dim = 10
filtr = ImportanceSelect(rfc,dim)
brX2 = filtr.fit_transform(brX,brY)
br2 = pd.DataFrame(np.hstack((brX2,np.atleast_2d(brY).T)))
cols = list(range(br2.shape[1]))
cols[-1] = 'Class'
br2.columns = cols
br2.to_csv('./RF/breast.csv')


# Abalone Dataset
abalone = pd.read_csv('./BASE/abalone.csv')
abaloneX = abalone.drop('Class', 1).copy().values
abaloneY = abalone['Class'].copy().values
abaloneX = StandardScaler().fit_transform(abaloneX)

cluster_range = range(1,11)
dims = range(1,10)

rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=-1)
fs_ab = rfc.fit(abaloneX,abaloneY).feature_importances_ 
tmp = pd.Series(np.sort(fs_ab)[::-1])
tmp.to_csv('./RF/abalone_scree.csv')

barplot_abalone(tmp)


dim = 6
filtr = ImportanceSelect(rfc,dim)
abaloneX2 = filtr.fit_transform(abaloneX,abaloneY)
abalone2 = pd.DataFrame(np.hstack((abaloneX2,np.atleast_2d(abaloneY).T)))
cols = list(range(abalone2.shape[1]))
cols[-1] = 'Class'
abalone2.columns = cols
abalone2.to_csv('./RF/abalone.csv')