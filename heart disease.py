#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[ ]:


dataset = pd.read_csv('heart.csv')


# In[ ]:


dataset.head()


# In[ ]:


dataset.columns


# In[ ]:


categorical = ['sex', 'cp', 'restecg', 'slope', 'thal']
do_not_touch = ['fbs', 'exang']
non_categorical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),categorical)],remainder='passthrough')
X = ct.fit_transform(dataset[categorical+do_not_touch+non_categorical])
y = dataset['target'].values


# In[ ]:


X[0,:]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[ ]:


X_train[:,-6:] = scaler.fit_transform(X_train[:,-6:])
X_test[:,-6:] = scaler.transform(X_test[:,-6:])


# In[ ]:


X_train[0,:]


# In[ ]:


from sklearn.svm import SVC
estimator = SVC()

parameters = [{'kernel':['rbf'],
               'C':[1,10,100,1000],
               'gamma':[1,0.1,0.001,0.0001],
            },
            {'kernel':['poly'],
               'C':[1,10,100,1000],
               'gamma':[1,0.1,0.001,0.0001],
             'degree':range(1,5)}
             ]


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'accuracy',
    n_jobs = 10,
    cv = 10,
    verbose=True
)


# In[ ]:


grid_search.fit(X_train, y_train)
grid_search.best_estimator_


# In[ ]:


y_pred = grid_search.best_estimator_.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
accuracy_score(y_test,y_pred)


# In[ ]:





# In[ ]:





# In[ ]:




