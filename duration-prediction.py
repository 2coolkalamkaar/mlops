#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('python -V')


# In[2]:


import pandas as pd
import os
import urllib.request

def download_data():
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    files = [
        ("green_tripdata_2021-01.parquet", "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet"),
        ("green_tripdata_2021-02.parquet", "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-02.parquet")
    ]
    
    for filename, url in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
        else:
            print(f"{filename} already exists.")

download_data()


# In[3]:


import pickle


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import root_mean_squared_error


# In[6]:


df = pd.read_parquet('./data/green_tripdata_2021-01.parquet')

df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

df = df[(df.duration >= 1) & (df.duration <= 60)]

categorical = ['PULocationID', 'DOLocationID']
numerical = ['trip_distance']

df[categorical] = df[categorical].astype(str)


# In[7]:


train_dicts = df[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)

target = 'duration'
y_train = df[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

print(f'Train RMSE: {root_mean_squared_error(y_train, y_pred)}')


# In[8]:


sns.distplot(y_pred, label='prediction')
sns.distplot(y_train, label='actual')

plt.legend()


# In[9]:


def read_dataframe(filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)

        df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
        df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


# In[10]:


df_train = read_dataframe('./data/green_tripdata_2021-01.parquet')
df_val = read_dataframe('./data/green_tripdata_2021-02.parquet')


# In[11]:


print(f'Train vs Val lengths: {len(df_train)}, {len(df_val)}')


# In[12]:


df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']


# In[13]:


categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
numerical = ['trip_distance']

dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)


# In[14]:


target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values


# In[15]:


lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

print(f'Validation RMSE (Linear Regression): {root_mean_squared_error(y_val, y_pred)}')


# In[16]:


with open('models/lin_reg.bin', 'wb') as f_out:
    pickle.dump((dv, lr), f_out)


# In[17]:


lr = Lasso(0.01)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

print(f'Validation RMSE (Lasso): {root_mean_squared_error(y_val, y_pred)}')


# In[ ]:




