
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
from collections import Counter 
from sklearn.ensemble import AdaBoostClassifier

####################### Load and analyze data  #######################

################# 1 load data
#Loading of the review
Review=pd.read_csv('reviews_32618_without_text.csv',delimiter=';', encoding = "ISO-8859-1")

#Loading of all the identities
Id=pd.read_csv('users_full_7034.csv',delimiter=';', encoding = "ISO-8859-1")

#Loading of all the personnalities
Perso=pd.read_csv('pers_scores_1098.csv',delimiter=';', encoding = "ISO-8859-1")

#Join to have one data base
df=Review.join(Perso.set_index('username'),on='username')
df=df.join(Id.set_index('username'),on='username')

################ 2 Chose the feature that will become the labels

# Rating
df['rating'].replace(1, 0,inplace=True)
df['rating'].replace(2, 0,inplace=True)
df['rating'].replace(3, 0,inplace=True)
df['rating'].replace(4, 1,inplace=True)
df['rating'].replace(5, 2,inplace=True)



################ 3  Data engineering

#Replace the instances of type by numbers
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



#Fill all the blank wich correspond to 0
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

# Create 3 area of location : Europe (=0) North America (=1) Asia and Australia (=2)

#Europe
df['location'].replace('London, United Kingdom', 0,inplace=True)
df['location'].replace('London', 0,inplace=True)
df['location'].replace('Essex', 0,inplace=True)
df['location'].replace('UK', 0,inplace=True)
df['location'].replace('Edinburgh, United Kingdom', 0,inplace=True)
df['location'].replace('Cumbria', 0,inplace=True)
df['location'].replace('Dublin, Ireland', 0,inplace=True)
df['location'].replace('Italia', 0,inplace=True)
df['location'].replace('Leeds', 0,inplace=True)
df['location'].replace('Southampton, United Kingdom', 0,inplace=True)
df['location'].replace('Italia', 0,inplace=True)
df['location'].replace('Wiltshire', 0,inplace=True)
df['location'].replace('Scotland', 0,inplace=True)
df['location'].replace('Aberdeen, United Kingdom', 0,inplace=True)
df['location'].replace('Birmingham, United Kingdom', 0,inplace=True)
df['location'].replace('South Wales, UK', 0,inplace=True)

#North America
df['location'].replace('Miami, FLorida', 1,inplace=True)
df['location'].replace('USA', 1,inplace=True)
df['location'].replace('Toronto, Canada', 1,inplace=True)
df['location'].replace('Los Angeles', 1,inplace=True)
df['location'].replace('California', 1,inplace=True)
df['location'].replace('Chicago, Illinois', 1,inplace=True)
df['location'].replace('New Mexico', 1,inplace=True)
df['location'].replace('Toronto', 1,inplace=True)
df['location'].replace('DC Metro Area', 1,inplace=True)
df['location'].replace('New York', 1,inplace=True)
df['location'].replace('US', 1,inplace=True)
df['location'].replace('New York City, New York', 1,inplace=True)
df['location'].replace('Phoenix, Arizona', 1,inplace=True)
df['location'].replace('NE US', 1,inplace=True)
df['location'].replace('Portland, Oregon', 1,inplace=True)
df['location'].replace('East Hartford', 1,inplace=True)
df['location'].replace('San Diego, CA', 1,inplace=True)
df['location'].replace('Wisconsin', 1,inplace=True)
df['location'].replace('Ohio', 1,inplace=True)
df['location'].replace('Fort Lauderdale, Florida', 1,inplace=True)
df['location'].replace('Los Angeles, California', 1,inplace=True)

#Asia and Australia
df['location'].replace('Hong Kong, China', 2,inplace=True)
df['location'].replace('Sydney', 2,inplace=True)
df['location'].replace('Melbourne, Australia', 2,inplace=True)
df['location'].replace('Sydney, Australia', 2,inplace=True)
df['location'].replace('Perth', 2,inplace=True)
df['location'].replace('Toronto', 1,inplace=True)

#Creation of a new feature
df['area']=nan
df.loc[df['location']==0, 'area']=0
df.loc[df['location']==1, 'area']=1
df.loc[df['location']==2, 'area']=2
#We fill the blank with the most common location we've see : Europe
df['area'].fillna(1,inplace=True)


#Creation of a new feature
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
df['taObjectCity'].replace('Singapore', 10,inplace=True)



df['travelcountry']=0
df.loc[df['taObjectCity'] == 0 , 'travelcountry'] = 0
df.loc[df['taObjectCity'] == 2 , 'travelcountry'] = 2
df.loc[df['taObjectCity'] == 1 , 'travelcountry'] = 1
df.loc[df['taObjectCity'] == 3 , 'travelcountry'] = 3
df.loc[df['taObjectCity'] == 4 , 'travelcountry'] = 4
df.loc[df['taObjectCity'] == 5 , 'travelcountry'] = 5
df.loc[df['taObjectCity'] == 7 , 'travelcountry'] = 7
df.loc[df['taObjectCity'] == 8 , 'travelcountry'] = 8
df.loc[df['taObjectCity'] == 9 , 'travelcountry'] = 9
df.loc[df['taObjectCity'] == 10 , 'travelcountry'] = 10



#Transform the date into a useable form
df['date'] = pd.to_datetime(df['date'])
#We create a new feature : the months
df['month'] = df['date'].dt.month


#Creations of new features using the Travel style

df['travelStyle'].fillna('RAS',inplace=True)

df['Family Hoilday Maker']=0
df.loc[df['travelStyle'].str.contains('Family Hoilday Maker'), 'Family Hoilday Maker']=0

df['Like a Local']=0
df.loc[df['travelStyle'].str.contains('Like a Local'), 'Like a Local']=1

                                                                                                                                                                                                                             
df['Luxury Traveller']=0
df.loc[df['travelStyle'].str.contains('Luxury Traveller'), 'Luxury Traveller']=1

df['Urban Explorer']=0
df.loc[df['travelStyle'].str.contains('Urban Explorer'), 'Urban Explorer']=1

df['Foodie']=0
df.loc[df['travelStyle'].str.contains('Foodie'), 'Foodie']=1

df['Vegetarian']=0
df.loc[df['travelStyle'].str.contains('Vegetarian'), 'Vegetarian']=1

df['Peace and Quiet Seeker']=0
df.loc[df['travelStyle'].str.contains('Peace and Quiet Seeker'), 'Peace and Quiet Seeker']=1

df['60+ Traveler']=0
df.loc[df['travelStyle'].str.contains('60+ Traveler'), 'area']=1

df['Art and Architecture Lover']=0
df.loc[df['travelStyle'].str.contains('Art and Architecture Lover'), 'Art and Architecture Lover']=1

df['Nature Lover']=0
df.loc[df['travelStyle'].str.contains('Nature Lover'), 'Nature Lover']=1



#We drop all the string features
df.drop('date',1,inplace=True)
df.drop('username',1,inplace=True)
df.drop('registerDate',1,inplace=True)
df.drop('taObjectCity',1,inplace=True)
df.drop('taObject',1,inplace=True)
df.drop('location',1,inplace=True)
df.drop('travelStyle',1,inplace=True)
df.drop('reviewerBadge',1,inplace=True)

#After a study this features seem to be unrelevant
df.drop('numPhotos',1,inplace=True)
df.drop('numRatings',1,inplace=True)




################ 4 We save it into a csv (to analyze it)
df.to_csv('df.csv',index = False)



################ 5 Creation of the training and testing sets

###### A Features
#  Training
X=df.iloc[:20000]

# Testing
Xtest=df.iloc[20001:]

###### B Labels
# Training
Y=X['rating']

# Testing
Ytest=Xtest['rating']

###### C Dropping the lables
X.drop('rating',1,inplace=True)
Xtest.drop('rating',1,inplace=True)



####################### Classify the data  #######################


###### LDA 
sklearn_lda = LDA()

## Fit the LDA
sklearn_lda = sklearn_lda.fit(X, Y)

## Transform the data
X_sklearn_lda=sklearn_lda.transform(X)
Xtest_sklearn_lda=sklearn_lda.transform(Xtest)

##### Classification Random forest
clf = RandomForestClassifier(n_estimators=500,max_features=0.7,max_depth=2,random_state=0)

#clf=AdaBoostClassifier(n_estimators=50,learning_rate=1)
## Fit the classifier
clf.fit(X_sklearn_lda, Y)


## Predict the class of X test
Y_predict=clf.predict(Xtest_sklearn_lda)
Y_predict=pd.DataFrame(Y_predict,columns=['Predicted Labels'])
score=clf.score(Xtest_sklearn_lda,Ytest)
print(score)


