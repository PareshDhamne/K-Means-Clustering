# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:31:25 2023

@author: PARESH DHAMNE

 Business Objective:

Minimize: Risk exposure for insurance companies by identifying high-risk policyholders.

Maximize: Customer satisfaction and retention through personalized insurance offerings.

Business Constraints:  
         Privacy and security of policyholder data.


DATA DICTIONARY:

'Customer': Unique identifier for each customer.
'State': The state where the customer resides.
'Customer Lifetime Value': The predicted net profit attributed to a customer over their entire relationship with the company.
'Response': Whether the customer responded to marketing initiatives.
'Coverage': The type of insurance coverage the customer has.
'Education': The highest level of education attained by the customer.
'Effective To Date': The date when the insurance policy becomes effective.
'EmploymentStatus': The employment status of the customer.
'Gender': The gender of the customer.
'Income': The annual income of the customer.
'Location Code': The type of area where the customer lives (urban, suburban, rural).
'Marital Status': The marital status of the customer.
'Monthly Premium Auto': The monthly premium amount for auto insurance.
'Months Since Last Claim': The number of months since the customer's last insurance claim.
'Months Since Policy Inception': The number of months since the inception of the insurance policy.
'Number of Open Complaints': The number of open complaints registered by the customer.
'Number of Policies': The number of insurance policies held by the customer.
'Policy Type': The type of insurance policy.
'Policy': The specific insurance policy.
'Renew Offer Type': The type of renewal offer provided to the customer.
'Sales Channel': The channel through which the insurance policy was sold.
'Total Claim Amount': The total amount claimed by the customer.
'Vehicle Class': The class of vehicle insured (e.g., SUV, sedan).
'Vehicle Size': The size of the insured vehicle (small, medium, large).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df=pd.read_csv("D:/Datasets/AutoInsurance.csv")
df

#EDA
#columns present in dataset
df.columns

df.shape

#5 number summary
df.describe()

#from this dataset education , customer and policy column is not important drop them
df.drop(['Education','Customer','Policy'],axis=1)

#generate dummy columns
df_new=pd.get_dummies(df)
df_new.columns
v=df_new.drop(df_new.loc[:,'Customer_AA10041':],axis=1)

# There is scale difference in columns so we normalize or standardized it
#Whenever there is mixed data we use normalizaion 

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#now apply this normlization function to Univ dataframe gor all the rows
df_norm=norm_fun(v)
df_norm
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
#shows elbow graph for given dataset

model=KMeans(n_clusters=3)

model.fit(df_norm)

model.labels_

mb=pd.Series(model.labels_)

#assign thsi series to df1 dataframe as column and name the column as "cluster
v['clust']=mb

v.head()

v.columns

#assign columns as sequence we want
df=v.iloc[:,:]

df.iloc[:,:].groupby(df.clust).mean()
#from the output cluster 2 has got highest Top 10
#lowest accept ratio,best faculty ratio and highest expenses
#highest graduates ratio

df.to_csv('Kmeans_autoinsurance.csv',encoding='utf-8')
import os
os.getcwd()    
