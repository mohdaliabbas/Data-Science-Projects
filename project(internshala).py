#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[64]:


data=pd.read_csv(r"project_train.csv")
data1=pd.read_csv(r"project_test.csv")


# In[65]:


data=data.eq('yes').mul(1)

data['subscribed'].value_counts()


# In[ ]:





# In[66]:


train=data[0:29999]
train


# In[67]:


test=data[30000:31647]
test


# In[68]:


x_train=train.drop('subscribed',axis=1)


# In[69]:


y_train=train['subscribed']
y_train


# In[70]:


x_test=test.drop('subscribed',axis=1)


# In[71]:


true_p=test['subscribed']
true_p


# In[72]:


from sklearn.linear_model import LogisticRegression 


# In[73]:


logreg=LogisticRegression()


# In[74]:


logreg.fit(x_train,y_train)


# In[75]:


pred=logreg.predict(x_test)


# In[76]:


pred


# In[77]:


#score function return the accuracy 
#test dataset acccuracy
#correct value of prediction(true_p)

logreg.score(x_test,true_p)


# In[78]:


#train dataset accuracy
logreg.score(x_train,y_train)


# In[62]:


#making  the test sample representative of train is called validation we cannot cover as of now


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




