import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#Importing users dataset
df_users = pd.read_csv('users.csv')
df_users = df_users.drop_duplicates(subset='username')

#Importing reviews dataset
df_reviews = pd.read_excel('reviews_32618_for_1098_users_with_location.xlsx')

#Reshape the reviews dataset, in order to have in rows users and in columns their ratings by city
df = df_reviews.pivot_table(index='username',columns=['taObjectCity'],values='rating',aggfunc=np.mean)

#Importing personnalities score and normalizing them so that each score has the same scale
df_pers = pd.read_excel('pers_scores_1098.xlsx',index_col = 'username')
df_pers = df_pers.subtract(df_pers.min(axis=0),axis=1).divide(df_pers.mean(axis=0)-df_pers.min(axis=0),axis=1)

#Rate each user that has at least 2 opinions common with the target user
def find_neighbours_by_opinion(username):
    dict_neighbours = {}
    df_common = df.loc[((df.loc[:,:]*df.loc[username,:]).count(axis='columns')>1),:]
    df_common = df_common.loc[:,df_common.loc[username,df_common.loc[username,:]>0].index]
    for val,row in df_common.iterrows():
        set_rated = df_common.loc[[val,username],(df_common.loc[val,:]*df_common.loc[username,:])>0].columns
        sim = np.sum(df_common.loc[val,set_rated]*df_common.loc[username,set_rated])/np.sqrt((np.sum(df_common.loc[val,set_rated]**2))*(np.sum(df_common.loc[username,set_rated]**2)))
        dict_neighbours[val] = sim
    return dict_neighbours

#Rate each user depending on the distance between personalities
def find_neighbours_by_personalities(username,dict_neighbours):
    df_pers_neighbours = df_pers.loc[dict_neighbours.keys(),:]
    df_pers_neighbours = (df_pers_neighbours.loc[username,:]*df_pers_neighbours).divide(np.sqrt((np.sum(df_pers_neighbours.loc[username,:]**2))*((df_pers_neighbours**2).sum(axis=1))),axis=0).sum(axis=1)
    return df_pers_neighbours

#Combine the two previous algorithms to obtain the closest neighbours
def find_k_closest_neighbours(username,k=10):
    dict_neighbours = find_neighbours_by_opinion(username)
    df_pers_neighbours = find_neighbours_by_personalities(username,dict_neighbours)
    df_sim = ((pd.Series(dict_neighbours) + df_pers_neighbours)/2).drop(username)
    sorted_df_sim = df_sim.sort_values(ascending=False)[:k]
    return sorted_df_sim

#Get top 3 destination recommended for an user
def get_recommandation(username,number_of_recommandation = 5):
    sorted_df_sim = find_k_closest_neighbours(username)
    neighbours = sorted_df_sim.index
    sims = sorted_df_sim.values
    condition = np.logical_or(df.loc[neighbours[1],:]>0,df.loc[neighbours[2],:]>0)
    for i in range(3,len(neighbours)-1):
        condition = np.logical_or(condition,df.loc[neighbours[i],:]>0)
    df_neighbours = df.loc[neighbours,condition]
    df_prediction = df.loc[username,:].mean() + df_neighbours.sub(df_neighbours.mean(axis='columns'),axis=0).multiply(sims,axis=0).sum().divide(df_neighbours.multiply(sims,axis=0).sum())
    df_predicted = df_prediction.sort_values(ascending=False,)[:number_of_recommandation]    
    print('The first destination recommended is ' + df_predicted.index[0]+ ', with a rating predicted of ' + str(df_predicted.values[0]))
    print('The second destination recommended is ' + df_predicted.index[1]+ ', with a rating predicted of ' + str(df_predicted.values[1]))
    print('The third destination recommended is ' + df_predicted.index[2]+ ', with a rating predicted of ' + str(df_predicted.values[2]))
    print('The fourth destination recommended is ' + df_predicted.index[3]+ ', with a rating predicted of ' + str(df_predicted.values[3]))
    print('The fifth destination recommended is ' + df_predicted.index[4]+ ', with a rating predicted of ' + str(df_predicted.values[4]))


