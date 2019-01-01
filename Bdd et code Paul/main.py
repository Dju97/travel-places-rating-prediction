
#All the import needed to use the sklearn function
from numpy import *
from sklearn import cross_validation
import csv as csv

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


####################### Load and treat data  #######################

################# 1 load data
#Loading of the review
Review=pd.read_csv('reviews_32618_without_text.csv',delimiter=';', encoding = "ISO-8859-1")

#Loading of all the identities
Id=pd.read_csv('users_full_7034.csv',delimiter=';', encoding = "ISO-8859-1")

#Loading of all the personnalities
Perso=pd.read_csv('pers_scores_1098.csv',delimiter=';', encoding = "ISO-8859-1")

#On fait un join des colonnes pour 
df=Review.join(Perso.set_index('username'),on='username')
df=df.join(Id.set_index('username'),on='username')
df.to_csv('df.csv',index = False)

X=df.iloc[:20000]
Xtest=df.iloc[20001:]


Y=X['rating']
X.drop('rating',1,inplace=True)
Xtest.drop('rating',1,inplace=True)

#######Treatement LDA 
#sklearn_lda = LDA()
#
### Fit the LDA
#sklearn_lda = sklearn_lda.fit(X, Y)
#
### Transform the data
#X_sklearn_lda=sklearn_lda.transform(X)
#Xtest_sklearn_lda=sklearn_lda.transform(Xtest)
#
###### Classification Random forest
#clf = RandomForestClassifier(n_estimators=500,max_features=0.7,max_depth=2,
#                             random_state=0)
#
#
### Fit the classifier
#clf.fit(X_sklearn_lda, Y)
#
#
### Predict the class of X test
#Y_predict=clf.predict(Xtest_sklearn_lda)







