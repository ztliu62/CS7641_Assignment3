import pandas as pd
import numpy as np
import os 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import load_breast_cancer

for d in ['BASE','RP','PCA','ICA','RF']:
    n = './{}/'.format(d)
    if not os.path.exists(n):
        os.makedirs(n)

OUT = './BASE/'

# Abalone Dataset
abalone = pd.read_csv('./Data/abalone.csv')

abalone.columns = ['Sex', 'Length', 'Diameter', 'Height',
					'Whole Weight', 'Shucked Weight', ' Viscera Weight',
					'Shell Weight', 'Ring']

le_sex = LabelEncoder()
abalone['Sex_encoded'] = le_sex.fit_transform(abalone.Sex)
Sex_ohe = OneHotEncoder()
X = Sex_ohe.fit_transform(abalone.Sex_encoded.values.reshape(-1,1)).toarray()
OneHot = pd.DataFrame(X, columns = ["Color_"+str(int(i)) for i in range(X.shape[1])])
abalone = pd.concat([abalone, OneHot], axis=1)
abalone = abalone.drop(['Sex', 'Sex_encoded'], axis = 1)

age_range = []
for i in abalone['Ring']:
    if i >= 1 and i <= 4:
        age_range.append('1')
    elif i >= 5 and i <= 8:
        age_range.append('2')
    elif i >= 9 and i <= 12:
        age_range.append('3')
    elif i >= 13 and i <= 16:
        age_range.append('4')
    elif i >= 17 and i <= 20:
        age_range.append('5')
    elif i >= 21 and i <= 24:
        age_range.append('6')
    elif i >= 25 and i <= 29:
        age_range.append('7')
abalone['Class'] = age_range
abalone = abalone.drop('Ring', axis = 1)

y = abalone.Class
X = abalone.drop('Class', axis=1)  

ab_data = pd.concat([X, y], axis = 1)

ab_data.to_csv(OUT+'abalone.csv', index=None) # with header

# Wine Quality Dataset
wineQuality = pd.read_csv('./Data/winequality-white.csv', ';')
winey = wineQuality.quality                 
wineX = wineQuality.drop('quality', axis=1)  

# Pre-processing of Dataset
# Merge Class - binary classification
winey1 = (winey > 5).astype(int)
Wine_data = pd.concat([wineX, winey1], axis = 1)
Wine_data.to_csv(OUT+'Wine.csv', index=None)

breast_cancers = load_breast_cancer(return_X_y = True)
brX, brY = breast_cancers
br = np.hstack((brX, np.atleast_2d(brY).T))
br = pd.DataFrame(br)
cols = list(range(br.shape[1]))
cols[-1] = 'Class'
br.columns = cols
br.to_csv(OUT+'breast.csv', index = None)



