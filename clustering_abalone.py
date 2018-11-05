import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import scale
from scipy.spatial.distance import cdist
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics.cluster import homogeneity_score, completeness_score

np.random.seed(42)

abalone = pd.read_csv('./BASE/abalone.csv')
abaloneX = abalone.drop('Class', 1).copy().values
abaloneY = abalone['Class'].copy().values
abaloneX = StandardScaler().fit_transform(abaloneX)

cluster_range = range(1,11)
# K-Means
def plot_kmeans(sse_before_dr, sse_after_dr, rm):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE of cluster')
    plt.title('Abalone Kmeans with ' + rm)

    plt.plot(cluster_range, sse_before_dr, color='firebrick', linestyle='-', marker='.', label="Clustering before DR")
    plt.plot(range(1, 11), sse_after_dr, color='olivedrab', linestyle='-', marker='.', label="Clustering after DR")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('./abalone_kmeans_'+rm+'.png')

def plot_sse(sse):
	ax = plt.figure().gca()
	plt.xlabel('Number of Clusters')
	plt.ylabel('SSE of cluster')
	plt.title('Abalone Clustering with Kmeans')

	plt.plot(cluster_range, sse, color='firebrick', linestyle='-', marker='.')
	plt.legend(loc='best')
	plt.grid()
	plt.savefig('./abalone_kmeans_sse.png')

def k_means(clusters, X):
    sse = []
    for k in clusters:
        estimator = KMeans(n_clusters=k, max_iter=500, init='k-means++', n_init=10)
        estimator.fit(X)
        min = np.min(np.square(cdist(X, estimator.cluster_centers_, 'euclidean')), axis=1)
        value = np.mean(min)
        sse.append(value)
    return sse

# Expectation Maximization
def plot_em(sse_before_dr, sse_after_dr, rm):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Number of clusters')
    plt.ylabel('Log Likelihood of cluster')
    plt.title('Abalone EM with ' + rm)

    plt.plot(cluster_range, sse_before_dr, color='firebrick', linestyle='-', marker='.', label="Clustering before DR")
    plt.plot(range(1, 11), sse_after_dr, color='olivedrab', linestyle='-', marker='.', label="Clustering after DR")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('./abalone_EM_'+rm+'.png')

def plot_ll(ll):
	ax = plt.figure().gca()
	plt.xlabel('Number of cluster')
	plt.ylabel('Log Likelihood')
	plt.title('Abalone Clustering with EM')

	plt.plot(cluster_range, ll, color = 'olivedrab', linestyle='-', marker='.')
	plt.legend(loc = 'best')
	plt.grid()
	plt.savefig('./abalone_em_ll.png')

def expectation_maximization(clusters, X):
    log_likelihood = []
    estimator = GMM(covariance_type='diag')
    for k in clusters:
        estimator.set_params(n_components=k)
        estimator.fit(X)
        log_likelihood.append(estimator.score(X))
    return log_likelihood

# General
def plot_score(clusters, X, Y, method):
	homo_score = []
	comp_score = []

	for k in clusters:
		if method == 'Kmeans':
			estimator = KMeans(n_clusters = k, max_iter=500, init='k-means++', n_init=10)
		elif method == 'GMM':
			estimator = GMM(n_components = k, covariance_type='diag')
		
		estimator.fit(X)
		homo_score.append(homogeneity_score(Y, estimator.predict(X)))
		comp_score.append(completeness_score(Y, estimator.predict(X)))

	ax = plt.figure().gca()
	plt.xlabel('Number of Cluster')
	plt.ylabel('Homogeneity/Completeness Scores')
	plt.title('Homogeneity/Completeness Scores of '+method)

	plt.plot(clusters, homo_score, color='firebrick', linestyle='-', marker='.', label="Homogeneity Score")
	plt.plot(clusters, comp_score, color='olivedrab', linestyle='-', marker='.', label="Completeness Score")
	
	plt.legend(loc = 'best')
	plt.grid()
	plt.savefig('./Abalone_'+method+'_scores.png')


sse_before_dr = k_means(cluster_range, abaloneX)
plot_sse(sse_before_dr)
plot_score(cluster_range, abaloneX, abaloneY, 'Kmeans')
ll_before_dr = expectation_maximization(cluster_range, abaloneX)
plot_ll(ll_before_dr)
plot_score(cluster_range, abaloneX, abaloneY, 'GMM')


# Compute PCA
abalone_pca_reduced = pd.read_csv('./PCA/abalone.csv')
abaloneX_pca = abalone_pca_reduced.drop('Class',1).copy().values
abaloneX_pca = StandardScaler().fit_transform(abaloneX_pca)

sse_after_dr = k_means(cluster_range, abaloneX_pca)
plot_kmeans(sse_before_dr, sse_after_dr, "PCA")
ll_after_dr = expectation_maximization(cluster_range, abaloneX_pca)
plot_em(ll_before_dr, ll_after_dr, 'PCA')

# Compute ICA
abalone_ica_reduced = pd.read_csv('./ICA/abalone.csv')
abaloneX_ica = abalone_ica_reduced.drop('Class',1).copy().values
abaloneX_ica = StandardScaler().fit_transform(abaloneX_ica)

sse_after_dr = k_means(cluster_range, abaloneX_ica)
plot_kmeans(sse_before_dr, sse_after_dr, "ICA")
ll_after_dr = expectation_maximization(cluster_range, abaloneX_ica)
plot_em(ll_before_dr, ll_after_dr, 'ICA')

# Compute RP
abalone_rp_reduced = pd.read_csv('./RP/abalone.csv')
abaloneX_rp = abalone_rp_reduced.drop('Class',1).copy().values
abaloneX_rp = StandardScaler().fit_transform(abaloneX_rp)

sse_after_dr = k_means(cluster_range, abaloneX_rp)
plot_kmeans(sse_before_dr, sse_after_dr, "RP")
ll_after_dr = expectation_maximization(cluster_range, abaloneX_rp)
plot_em(ll_before_dr, ll_after_dr, 'RP')

# Compute RF
abalone_rf_reduced = pd.read_csv('./RF/abalone.csv')
abaloneX_rf = abalone_rf_reduced.drop('Class',1).copy().values
abaloneX_rf = StandardScaler().fit_transform(abaloneX_rf)

sse_after_dr = k_means(cluster_range, abaloneX_rf)
plot_kmeans(sse_before_dr, sse_after_dr, "RF")
ll_after_dr = expectation_maximization(cluster_range, abaloneX_rf)
plot_em(ll_before_dr, ll_after_dr, 'RF')
