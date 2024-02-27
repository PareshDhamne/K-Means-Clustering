# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:18:19 2023

@author: PARESH DHAMNE

Business Objective:

Minimize: Operational costs associated with acquiring new customers to replace churned ones.

Maximize: Revenue from long-term customer relationships and increased customer lifetime value.

Business Constraints:  
     Data quality and availability for analysis, especially regarding historical churn data.
"""
'''
The data is a mixture of both categorical and numerical data.
It consists of the number of customers who churn out.
Derive insights and get possible information on factors that
may affect the churn decision. Refer to Telco_customer_churn.xlsx dataset.
This sample data module tracks a fictional telco company's customer churn based 
on various factors.T he churn column indicates whether the customer
departed within the last month. 
Other columns include gender, dependents, monthly charges,
and many with information about the types 
of services each customer has.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df=pd.read_excel("D:/Datasets/Telco_customer_churn.xlsx")
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

# we are checking which column is not necserray or
# the column which is  numerical data  that can be place in the datframe
df1=df.drop(['Customer ID','Count', 'Quarter', 'Referred a Friend', 'Offer', 'Phone Service',
       'Multiple Lines','Internet Service', 'Internet Type', 'Online Security', 
       'Online Backup', 'Device Protection Plan','Premium Tech Support', 'Streaming TV',
       'Streaming Movies','Streaming Music', 'Unlimited Data', 'Contract',
       'Paperless Billing','Payment Method'],axis=1) 

df1.columns

df1.shape

# There is scale difference in columns so we normalize or standardized it
#Whenever there is mixed data we use normalizaion 

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#now apply this normlization function to Univ dataframe gor all the rows
df_norm=norm_fun(df1.iloc[:,:])
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

#assign thsi series to df1 dataframe as column
df1['clust']=mb

df=df1.iloc[:,[10,1,2,3,4,5,6,7,8,9]]

df.iloc[:,:].groupby(df.clust).mean()

#from the output cluster 2 has got highest Top 10
#lowest accept ratio,best faculty ratio and highest expenses
#highest graduates ratio

df.to_csv('Kmeans_telco_churn_case.csv',encoding='utf-8')
import os
os.getcwd()    
