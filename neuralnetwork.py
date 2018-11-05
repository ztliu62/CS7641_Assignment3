import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import csv

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import nn_arch,nn_reg, myGMM
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from helpers import ImportanceSelect
from sklearn.ensemble import RandomForestClassifier

def plot_nn_dm_analysis():
    results = pd.read_csv('./NN_results.csv')

    original_accuracy = results['Benchmark']
    pca_accuracy = results['PCA']
    ica_accuracy= results['ICA']
    rp_accuracy = results['RP']
    rf_accuracy = results['RF']

    n_dimensions = range(1, 11)

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(n_dimensions, original_accuracy, color='royalblue', linewidth=2, linestyle='-', marker='p', label="Benchmark")
    plt.plot(n_dimensions, pca_accuracy, color='firebrick', linestyle='-', marker='.', label="PCA")
    plt.plot(n_dimensions, ica_accuracy, color='olivedrab', linestyle='-', marker='.', label="ICA")
    plt.plot(n_dimensions, rp_accuracy, color='blueviolet', linestyle='-', marker='.', label="RP")
    plt.plot(n_dimensions, rf_accuracy, color='darkorange', linestyle='-', marker='.', label="RF")

    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(50, 95, 5))
    plt.legend(loc='lower right')
    plt.xlabel("# of Dimensions")
    plt.ylabel("Accuracy(%)")
    plt.title("Neural Networks Dimension Reduction - Accuracy")
    plt.grid()
    plt.savefig('./neuralnetwork_dr_acc.png')

np.random.seed(42)
dims = range(1,11)

wine = pd.read_csv('./BASE/Wine.csv')
wineX = wine.drop('quality', 1).copy().values
wineY = wine['quality'].copy().values
wineX= StandardScaler().fit_transform(wineX)

# Benchmark 
grid ={'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=1000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(wineX,wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv('./BASE/NN_bmk.csv')
benchmark = gs.best_score_
results = defaultdict(dict)

for i in dims:
	results[i]['Benchmark'] = 100.*benchmark
	# PCA
	grid ={'pca__n_components':[i] ,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
	pca = PCA(random_state=5)       
	mlp = MLPClassifier(activation='relu',max_iter=1000,early_stopping=True,random_state=5)
	pipe = Pipeline([('pca',pca),('NN',mlp)])
	gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

	gs.fit(wineX, wineY)
	results[i]['PCA'] = 100.*gs.best_score_
	tmp = pd.DataFrame(gs.cv_results_)
	tmp.to_csv('./PCA/nn.csv')

	# ICA
	grid ={'ica__n_components':[i],'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
	ica = FastICA(random_state=5)       
	mlp = MLPClassifier(activation='relu',max_iter=1000,early_stopping=True,random_state=5)
	pipe = Pipeline([('ica',ica),('NN',mlp)])
	gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

	gs.fit(wineX, wineY)
	results[i]['ICA'] = 100.*gs.best_score_
	tmp = pd.DataFrame(gs.cv_results_)
	tmp.to_csv('./ICA/nn.csv')

	# RP
	grid ={'rp__n_components':[i],'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
	rp = SparseRandomProjection(random_state=5)       
	mlp = MLPClassifier(activation='relu',max_iter=1000,early_stopping=True,random_state=5)
	pipe = Pipeline([('rp',rp),('NN',mlp)])
	gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

	gs.fit(wineX,wineY)
	results[i]['RP'] = 100.*gs.best_score_
	tmp = pd.DataFrame(gs.cv_results_)
	tmp.to_csv('./RP/nn.csv')


	rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=-1)
	filtr = ImportanceSelect(rfc)
	grid ={'filter__n':[i],'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
	mlp = MLPClassifier(activation='relu',max_iter=1000,early_stopping=True,random_state=5)
	pipe = Pipeline([('filter',filtr),('NN',mlp)])
	gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

	gs.fit(wineX, wineY)
	results[i]['RF'] = 100.*gs.best_score_
	tmp = pd.DataFrame(gs.cv_results_)
	tmp.to_csv('./RF/nn.csv')

results = pd.DataFrame(results).T
results.to_csv('./NN_results.csv')

plot_nn_dm_analysis()

'''
clusters =  [1,2,3,4,5,6,7,8,9,10]

grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=1000,early_stopping=True,random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km',km),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(wineX,wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv('./wine_cluster_Kmeans.csv')

grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=1000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(wineX,wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv('./wine_cluster_GMM.csv')
'''