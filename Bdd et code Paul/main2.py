
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

df.drop('date',1,inplace=True)
df.drop('username',1,inplace=True)
df.drop('registerDate',1,inplace=True)
#df.drop('taObjectCity',1,inplace=True)
df.drop('taObject',1,inplace=True)
df.drop('location',1,inplace=True)
df.drop('travelStyle',1,inplace=True)
df.drop('reviewerBadge',1,inplace=True)
df.drop('rating',1,inplace=True)

df['type'].replace('Hotels', 0,inplace=True)
df['type'].replace('Restaurants', 1,inplace=True)
df['type'].replace('Attractions', 2,inplace=True)


#Transform Sex feature into a digital one
df['gender'].replace('female', 0,inplace=True)
df['gender'].replace('male', 1,inplace=True)
df['gender'].fillna(round(df['gender'].mean()),inplace=True)

#Transform agerange into a float
df['ageRange'].replace('18-24', 0,inplace=True)
df['ageRange'].replace('25-34', 1,inplace=True)
df['ageRange'].replace('25-34', 2,inplace=True)
df['ageRange'].replace('35-49', 3,inplace=True)
df['ageRange'].replace('50-64', 4,inplace=True)
df['ageRange'].replace('65+', 5,inplace=True)
df['ageRange'].fillna(round(df['ageRange'].mean()),inplace=True)


#df['rating'].replace(1, 0,inplace=True)
#df['rating'].replace(2, 0,inplace=True)
#df['rating'].replace(3, 1,inplace=True)
#df['rating'].replace(4, 1,inplace=True)
#df['rating'].replace(5, 1,inplace=True)



df['helpfulness'].fillna(0,inplace=True)
df['numHotelsReviews'].fillna(0,inplace=True)
df['numRestReviews'].fillna(0,inplace=True)
df['numAttractReviews'].fillna(0,inplace=True)
df['numFirstToReview'].fillna(0,inplace=True)
df['numRatings'].fillna(0,inplace=True)
df['numPhotos'].fillna(0,inplace=True)
df['numForumPosts'].fillna(0,inplace=True)
df['numArticles'].fillna(0,inplace=True)
df['numCitiesBeen'].fillna(0,inplace=True)
df['totalPoints'].fillna(0,inplace=True)
df['contribLevel'].fillna(0,inplace=True)
df['numHelpfulVotes'].fillna(0,inplace=True)

df=df.loc[df['taObjectCity'].isin(['London','New York City','Paris','Las Vegas','Chicago','San Francisco','Singapore','Rome','Los Angeles','Orlando','Sidney'])]
df['taObjectCity'].replace('London', 0,inplace=True)
df['taObjectCity'].replace('New York City', 1,inplace=True)
df['taObjectCity'].replace('Paris', 2,inplace=True)
df['taObjectCity'].replace('Las Vegas', 3,inplace=True)
df['taObjectCity'].replace('Chicago', 4,inplace=True)
df['taObjectCity'].replace('San Francisco', 5,inplace=True)
df['taObjectCity'].replace('Rome', 6,inplace=True)
df['taObjectCity'].replace('Los Angeles', 7,inplace=True)
df['taObjectCity'].replace('Orlando', 8,inplace=True)
df['taObjectCity'].replace('Sidney', 9,inplace=True)
df['taObjectCity'].replace('Singapore', 9,inplace=True)

df['travelcountry']=0
df.loc[df['taObjectCity'] == 0 , 'travelcountry'] = 0

df['samecity']=0


df.to_csv('df.csv',index = False)




X=df.iloc[:3000]
Xtest=df.iloc[3001:]


Y=X['taObjectCity']
Ytest=Xtest['taObjectCity']
X.drop('taObjectCity',1,inplace=True)
Xtest.drop('taObjectCity',1,inplace=True)



######Treatement LDA 
sklearn_lda = LDA()

## Fit the LDA
sklearn_lda = sklearn_lda.fit(X, Y)

## Transform the data
X_sklearn_lda=sklearn_lda.transform(X)
Xtest_sklearn_lda=sklearn_lda.transform(Xtest)

##### Classification Random forest
clf = RandomForestClassifier(n_estimators=500,max_features=0.7,max_depth=2,
                             random_state=0)


## Fit the classifier
clf.fit(X, Y)


## Predict the class of X test
#Y_predict=clf.predict(Xtest)
score=clf.score(Xtest,Ytest)







