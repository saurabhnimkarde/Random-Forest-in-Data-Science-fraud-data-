#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[3]:


fraud = pd.read_csv("Fraud_check.csv")


# In[4]:


fraud.head()


# In[5]:


fraud.shape


# In[6]:


fraud.info()


# In[7]:


fraud.rename({'Undergrad':'UG','Marital.Status':'MS', 'Taxable.Income':'TI', 'City.Population':'CP', 'Work.Experience':'WE'},axis = 1, inplace = True)


# In[8]:


fraud.head()


# In[9]:


fraud['TI'] = fraud.TI.map(lambda taxable_income : 'Risky' if taxable_income <= 30000 else 'Good')


# In[10]:


fraud.head()


# In[11]:


fraud['UG'] = fraud['UG'].astype("category")
fraud['MS'] = fraud['MS'].astype("category")
fraud['Urban'] = fraud['Urban'].astype("category")
fraud['TI'] = fraud['TI'].astype("category")


# In[12]:


fraud.dtypes


# In[13]:


label_encoder = preprocessing.LabelEncoder()
fraud['UG'] = label_encoder.fit_transform(fraud['UG'])

fraud['MS'] = label_encoder.fit_transform(fraud['MS'])

fraud['Urban'] = label_encoder.fit_transform(fraud['Urban'])

fraud['TI'] = label_encoder.fit_transform(fraud['TI'])


# In[14]:


fraud


# In[15]:


fraud['TI'].unique()


# In[16]:


fraud['TI'].value_counts()


# In[17]:


X = fraud.iloc[:,[0,1,3,4,5]]
Y = fraud.iloc[:,2]


# In[18]:


X


# In[19]:


Y


# In[20]:


x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.3, random_state = 10)


# In[21]:



from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,max_features=4,criterion="entropy")


# In[48]:





# In[22]:


rf.fit(x_train,y_train) 
rf.estimators_  
rf.classes_ 
rf.n_classes_ 
rf.n_features_  

rf.n_outputs_ 


# In[23]:


preds = rf.predict(x_test)
preds


# In[24]:


pd.Series(preds).value_counts()


# In[25]:


crosstable = pd.crosstab(preds,y_test)
crosstable


# In[26]:


np.mean(preds==y_test)


# In[27]:


print(classification_report(preds,y_test))


# In[ ]:




