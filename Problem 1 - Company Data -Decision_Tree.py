#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder#for encoding
from sklearn.model_selection import train_test_split#for train test splitting
from sklearn.tree import DecisionTreeClassifier#for decision tree object
from sklearn.metrics import classification_report, confusion_matrix#for checking testing results
from sklearn.tree import plot_tree#for visualizing tree 


# In[2]:


#reading the data
df = pd.read_csv('Company_Data.csv')
df.head()


# In[3]:


#getting information of dataset
df.info()


# In[4]:


df.shape


# In[5]:


df.isnull().any()


# In[6]:


# let's plot pair plot to visualise the attributes all at once
sns.pairplot(data=df, hue = 'ShelveLoc')


# In[7]:


#Creating dummy vairables dropping first dummy variable
df=pd.get_dummies(df,columns=['Urban','US'], drop_first=True)


# In[8]:


df


# In[9]:


df.info()


# In[10]:


from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# In[11]:


df['ShelveLoc']=df['ShelveLoc'].map({'Good':1,'Medium':2,'Bad':3})


# In[12]:


df.head()


# In[13]:


x=df.iloc[:,0:6]
y=df['ShelveLoc']


# In[14]:


x


# In[15]:


y


# In[16]:


df['ShelveLoc'].unique()


# In[17]:


df.ShelveLoc.value_counts()


# In[18]:


colnames = list(df.columns)
colnames


# In[19]:


# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# # Building Decision Tree Classifier using Entropy Criteria

# In[20]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[21]:


from sklearn import tree


# In[22]:


#PLot the decision tree
tree.plot_tree(model);


# In[23]:


fn=['Sales','CompPrice','Income','Advertising','Population','Price']
cn=['1', '2', '3']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[24]:


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category


# In[25]:


preds


# In[26]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[27]:


# Accuracy 
np.mean(preds==y_test)


# # Building Decision Tree Classifier (CART) using Gini Criteria

# In[28]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[29]:


model_gini.fit(x_train, y_train)


# In[30]:


#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)


# # Decision Tree Regression Example

# In[31]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor


# In[32]:


array = df.values
X = array[:,0:3]
y = array[:,3]


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[34]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[35]:


#Find the accuracy
model.score(X_test,y_test)


# In[ ]:




