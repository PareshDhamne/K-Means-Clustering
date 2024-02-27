# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:30:57 2023

@author: Paresh Dhamne

Business Objective:
minimize:Customer dissatisfaction and churn rates by identifying and addressing pain points in services.
maximize: Revenue from additional services such as frequent flyer programs, premium seat bookings, and ancillary services.

Business Constraints:
                 Regulatory compliance with safety standards and industry regulations.
"""
'''
ID#		                Unique ID
Topflight		      	Indicates whether flyer has attained elite "Topflight" status, 1 = yes, 0 = no
Balance		          	Number of miles eligible for award travel
Qual_miles	          	Number of miles counted as qualifying for Topflight status
cc1_miles		      	Has member earned miles with airline freq. flyer credit card in the past 12 months (1=Yes/0=No)?
cc2_miles		      	Has member earned miles with Rewards credit card in the past 12 months (1=Yes/0=No)?
cc3_miles		      	Has member earned miles with Small Business credit card in the past 12 months (1=Yes/0=No)?
Bonus_miles		      	Number of miles earned from non-flight bonus transactions in the past 12 months
Bonus_trans				Number of non-flight bonus transactions in the past 12 months
Flight_miles_12mo		Number of flight miles in the past 12 months
Flight_trans_12 	    Number of flight transactions in the past 12 months
Day_since_enroll        Date of enroll  for flight
Award?                  Number of awards won by flight
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df=pd.read_excel("D:/Datasets/EastWestAirlines.xlsx")
df

#EDA
#columns present in dataset
df.columns

#display number of rows and columns
df.shape

#5 number summary
df.describe()

#In dataset ID# and Award? are not important so drop it

df1=df.drop(['ID#','Award?'],axis=1)

df1.columns

#display the number of rows and columns
df1.shape

#check any null values is there in dataset
df1.isnull()

# There is scale difference in columns so we normalize or standardized it
#Whenever there is mixed data we use normalizaion 

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#now apply this normlization function to Univ dataframe gor all the rows

df_norm=norm_fun(df1.iloc[:,:])
#it will give us normalize value between 1 to 0

TWSS=[]
k=list(range(1,10))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
TWSS
#as k value increases the TWSS value decreases
plt.plot(k,TWSS,'ro-');
plt.xlabel("No_of_clusters");
plt.ylabel("Total_within_SS");


model=KMeans(n_clusters=3)

model.fit(df_norm)

model.labels_

mb=pd.Series(model.labels_)

#assign thsi series to df1 dataframe as column and name the column as "cluster
df1['clust']=mb

df1.head()

#assign columns as sequence we want
df=df1.iloc[:,[3,1,2,4,5,6,7,8,9,10]]

df.iloc[:,2:].groupby(df.clust).mean()
#from the output cluster 2 has got highest Top 10
#lowest accept ratio,best faculty ratio and highest expenses
#highest graduates ratio

df.to_csv('Kmeans_airlines.csv',encoding='utf-8')
import os
os.getcwd()    






























































































































































































































































































































































































































































































































































































































