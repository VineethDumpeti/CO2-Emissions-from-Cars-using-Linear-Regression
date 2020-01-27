#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df = pd.read_csv(r"C:\Users\Admin\Downloads\FuelConsumptionCo2.csv")


# In[9]:


df.head(5)


# In[10]:


df.describe()


# In[11]:


df = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]


# In[12]:


df.head(5)


# In[15]:


viz = df[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()


# In[19]:


plt.scatter(df.FUELCONSUMPTION_COMB, df.CO2EMISSIONS, color = "green")
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("EMISSION")
plt.show


# In[21]:


plt.scatter(df.ENGINESIZE, df.CO2EMISSIONS, color = "blue")
plt.xlabel("ENGINESIZE")
plt.ylabel("EMISSION")
plt.show()


# In[24]:


plt.scatter(df.CYLINDERS, df.CO2EMISSIONS, color = "red")
plt.xlabel("CYLINDERS")
plt.ylabel("EMISSION")
plt.show()


# In[25]:


msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[msk]


# In[26]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[27]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[28]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[29]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )


# In[ ]:




