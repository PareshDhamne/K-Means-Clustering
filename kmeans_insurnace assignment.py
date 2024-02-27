# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 00:10:19 2023

@author: PARESH DHAMNE
"""

'''
given dataset is based on insurance paid by and climed by customer by different age
first column 'Premiums Paid' gives the inforamtion about primiumns paid by customer
second column' Age' gives the age of the person for each primimum paid
third column 'Days to Renew' gives the remaining number of days for renew of insurance
fourth column 'Claims made' gives the average claims made by customer from different age
fifth column 'Income' gives the income


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df=pd.read_csv("D:/Datasets/Insurance Dataset.csv")
df

#EDA
#columns present in dataset
df.columns

#display number of rows and columns
df.shape

#check any null values is there in dataset
df.isnull()

#5 number summary
df.describe()

#except 1st all columns are important so we are not dropping any column

# There is scale difference in columns so we normalize or standardized it
#Whenever there is mixed data we use normalizaion 

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#now apply this normlization function to Univ dataframe gor all the rows
df_norm=norm_fun(df.iloc[:,:])
#it will give us normalize value between 1 to 0

TWSS=[]
k=list(range(1,6))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
TWSS
#as k value increases the TWSS value decreases
plt.plot(k,TWSS,'ro-');
plt.xlabel("No_of_clusters");
plt.ylabel("Total_within_SS");
#shows elbow graph for given dataset

model=KMeans(n_clusters=3)

model.fit(df_norm)

model.labels_

mb=pd.Series(model.labels_)

#assign thsi series to df1 dataframe as column and name the column as "cluster
df['clust']=mb

df.head()

#assign columns as sequence we want
df=df.iloc[:,[1,2,3,4,5]]

df.iloc[:,:].groupby(df.clust).mean()
#from the output cluster 2 has got highest Top 10
#lowest accept ratio,best faculty ratio and highest expenses
#highest graduates ratio

df.to_csv('Kmeans_insurance.csv',encoding='utf-8')
import os
os.getcwd()    
