This file is the README for CS7641 Assignment 3

File Structure of the package is as follows:
- ztliu62-analysis.pdf: detailed analysis of the unsupervised learning and dimensionality reduction algorithms
- README.txt: instruction and link for running code


Code Link: 
https://github.com/ztliu62/CS7641_Assignment3

Code Folder:
- /Data:
	- abalone.csv: raw data of abalone dataset
	- winequality-white.csv: raw data of wine quality dataset
-parse.py: pre-processing raw data and parse data for abalone/wine/breastcancer dataset
- PCA.py: Code for PCA algorithm
- ICA.py: Code for ICA algorithm
- RP.py: Code for Random Projection algorithm
- RF.py: Code for Random Forest algorithm
- helpers.py: Code for auxiliary use
- clustering_abalone.py: Code for clustering algorithms on abalone dataset
- clustering_breast.py: Code for clustering algorithm on breast-cancer dataset
- neuralnetwork.py: Code for neural network optimization on wine datatset

Code Execution Instruction: 
1. Run parse.py to generate the data for further processing. It'll also generate the appropriate folder structure to store data
2. Run PCA.py, ICA.py, RP.py ad RF.py to generate dimensionality reduction data
3. Run clustering_abalone.py and clustering_breast.py to execute K-means and EM algorithms on both original data and the generated reduced data
4. Run neuralnetwork.py to get both the results of applying dimensionality reduction and clustering on neural network. 

The code was written and configured in Python 3.7, sklearn 0.19.1, numpy 1.14.5 and pandas 0.23.3