#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib.pyplot as plt
import pandas_datareader as preader
import quandl
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import metrics
from sklearn import preprocessing


# In[2]:


#Import the datset from CSV file into pandas DataFrame
#NOTE: PLEASE CHANGE THE PATH OF DATASET TO FOLDER WHERE DATASET RESIDES ON COMPUTER USED TO EXECUTE THE CODE
df = pd.read_csv('/Users/gayatrihungund/Downloads/EOD-DIS.csv')
df.set_index('Date',inplace=True)
#Statistical Analysis of Walt Disney Stock Dataset
df.describe()


# In[3]:


#A subset of data is displayed to see the various values in a dataset
#After selecting data it can be noticed that the entries for "2019-05-11" and "2017-05-12" are missing
#The reason behind the missing values is that the New York Stock exchange is closed on weekend and public holidays
df.head()


# In[4]:


#querying dataset to get date with lowest stock price
print(df.loc[df['Low'] == 13.48])


# In[5]:


#Querying dataset to get stock prices higher price values than mean
print(df.loc[df['High'] >= 62.738591])


# In[6]:


#get_ipython().run_line_magic('matplotlib', 'inline')
#The frequency distribution will help to study how the data is spread. 

#The frequency distribution of Open price shows that the frequency of prices in approximate range 35 to 50
#are most popular whereas when other factors such as dividend and split are taken into consideration 
#while calculating the adjusted open price of stock this range changes to approximately 22 to 28.

#While trading for next day or investing on the stock on next day, the variation in the adjusted and real prices
#of stock should be studied as the adjusted prices are more accurate as they consider various factors 
#after the stock market closes for the day.
plt.figure(figsize=(15, 5))
plt.title('Distribution of Open Price and Adjusted Open Price')
plt.xlabel('Open Price and Adjusted Open Price')
plt.ylabel('Count')
plt.hist([df['Open'],df['Adj_Open']], bins='auto')
plt.legend(['Open Price','Adjusted Open Price'])


# In[7]:


#Frequency distribution of high price and adjusted high price is plotted as follows
plt.figure(figsize=(15, 5))
plt.title('Distribution of High Price and Adjusted High Price')
plt.xlabel('High Price and Adjusted High Price')
plt.ylabel('Count')
plt.hist([df['High'],df['Adj_High']], bins='auto')
plt.legend(['High Price','Adjusted High Price'])


# In[8]:


#Frequency distribution of low price and adjusted low price is plotted as follows
plt.figure(figsize=(15, 5))
plt.title('Distribution of Low Price and Adjusted Low Price')
plt.xlabel('Low Price and Adjusted Low Price')
plt.ylabel('Count')
plt.hist([df['Low'],df['Adj_Low']], bins='auto')
plt.legend(['Low Price','Adjusted Low Price'])


# In[9]:


#Frequency distribution of volume and adjusted volume is plotted as follows
plt.figure(figsize=(15, 5))
plt.title('Distribution of Volume and Adjusted Volume')
plt.xlabel('Volume and Adjusted Volume')
plt.ylabel('Count')
plt.hist([df['Volume'],df['Adj_Volume']], bins='auto')
plt.legend(['Volume','Adjusted Volume'])


# In[10]:


#Frequency distribution of split is plotted as follows
plt.figure(figsize=(15, 5))
plt.title('Distribution of Split')
plt.xlabel('Split')
plt.ylabel('Count')
plt.hist(df['Split'], bins='auto')
plt.legend(['Split'])


# In[11]:


#Frequency distribution of dividend is plotted as follows
plt.figure(figsize=(15, 5))
plt.title('Distribution of Dividend')
plt.xlabel('Dividend')
plt.ylabel('Count')
plt.hist(df['Dividend'], bins='auto')
plt.legend(['Dividend'])


# In[12]:


#The relationship between the dataframe features can be determined by studying the bivariate 
#frequency distribution created by using pairplot.

#Consider the relation between the open and high the scatter plot is almost linear which says that as open price increases,
#the high price will also increase.
sns.pairplot(df)


# In[13]:


#check if the data frames contain any values that are null
print(df.isna().sum())
df = df.dropna(inplace=False)


# In[16]:


#To check the co-relation between the stock market dataset features we have used correlation matrix
#As per the correlation matrix, we can decide if the value of selected features decrease or increase on change in 
#some other feature. 

#For example, the value of correlation coefficient is -0.24 for Volume and Open which means that 
#as the open price of stock increases the number of investors will decrease.

#The matrix also demonstrates high correlation between the Open and High price of stock.

#Also the split has negative or almost zero correlation with the other features of stock price data
#which means that the variable has no relation or extremely weak relation with other features
#Further considering the dividend also has weak relation with other features.

df1 = df[['Open','High','Low','Close','Volume','Dividend','Split','Adj_Open','Adj_High','Adj_Low','Adj_Volume']]
cor = df1.corr()
cor


# In[17]:


#Removing the target variable for prediction
y = df.pop("Close")
print(y)
adj_y = df.pop("Adj_Close")


# In[18]:


#Splitting the data into training and test dataset.
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=1)
print("Training Set:")
print(X_train)
print("Closing Price of Stock in Training Set")
print(y_train)


# In[19]:


#Printing data before normalization
print(df)


# In[20]:


#Checking the statistical distribution of data before normalizing the data 
#will help to know the minimum and maximum in the data.

train_stats = X_train.describe()
print(train_stats)

#Conversion of index to date time format to make it easier to plot graphs
df3 = pd.DataFrame()
df3 = X_test
dates = pd.to_datetime(df3.index)

#MinMax Scaling
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler = min_max_scaler.fit(X_train)
df_train_norm = min_max_scaler.transform(X_train)


min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler = min_max_scaler.fit(X_test)
df_test_norm = min_max_scaler.transform(X_test)


# In[21]:


#Data after normalization
print(df_train_norm)


# In[22]:


#Test Case 1: When the max depth is set to values in range 1 to 10, the MSE is high and about 2.97
#Test Case 2: When the max depth is set to values in range > 10, the MSE is high and about 2.71
#Test Case 3: The n_estimators set to 49 gives the best MSE value of 2.6629 and the values greater than 59 decrease the MSE to 2.661
#Test Case 4: As a multicore machine is used to execute the code the value of n_jobs is set to -1 to use all cores
warnings.filterwarnings("ignore")
print("\n\nRandomForestRegressor")
random_forest =  RandomForestRegressor(random_state=1,n_estimators=59,max_depth=50,bootstrap=True,n_jobs=-1)
random_forest.fit(df_train_norm,y_train)
y_randomForestPred = random_forest.predict(df_test_norm)
randomForestMSE = mean_squared_error(y_test,y_randomForestPred)
print("MSE      = " + str(randomForestMSE))
randomForestR2 = r2_score(y_test,y_randomForestPred)
print("r2_score = " + str(randomForestR2))
#print("Explained Variance Score = " + str(explained_variance_score(y_test,y_randomForestPred)))
plt.figure(figsize=(20,20))
plt.scatter(dates,y_randomForestPred,label="Predicted Closing Stock Price")
plt.scatter(dates,y_test,c="#ADD8E6",label="Actual Closing Stock Price")
plt.title("Predicted Closing Price of Stock and Actual Closing Price of Stock using Random Forest Regressor",fontsize=20)
plt.xlabel('Time in Years',fontsize=20)
plt.ylabel('Stock Closing Price Predictions(Random Forest Regressor)',fontsize=20)
plt.legend(loc='best')
plt.show()


# In[23]:


#Test Case 1: The value of learning_rate is first set to 0.001 then the MSE obtained is 2.58
#Test Case 2: The value of learning_rate is first set to 0.001 then the MSE obtained is 2.5819
#Test Case 3: The value of learning_rate is first set to 0.000167 then the MSE obtained is 2.580
print("AdaBoost Regressor")
regr_ada = AdaBoostRegressor(random_forest,n_estimators=30,random_state=1,learning_rate=0.000167)
regr_ada.fit(df_train_norm,y_train)
regr_ada_pred = regr_ada.predict(df_test_norm)
acc_ada_mse = mean_squared_error(y_test,regr_ada_pred)
print("MSE      = "+ str(acc_ada_mse))
r2_ada = r2_score(y_test,regr_ada_pred)
print("r2_score = " + str(r2_ada))
plt.figure(figsize=(20,20))
plt.scatter(dates,regr_ada_pred,label="Predicted Closing Stock Price")
plt.scatter(dates,y_test,c="orange",label="Actual Closing Stock Price")
plt.title("Predicted Closing Price of Stock and Actual Closing Price of Stock using ADA Boost Regressor",fontsize=20)
plt.xlabel('Time in Years',fontsize=20)
plt.ylabel('Stock Closing Price Predictions(AdaBoost-RandomForest Regressor)',fontsize=20)
plt.legend(loc='best')
plt.show()


# In[24]:


#Test Case 1: The Best MSE obtained using Linear Regression is 5.80
print("Linear Regression")
lin = linear_model.LinearRegression(n_jobs=-1)
fit_data = lin.fit(df_train_norm,y_train)
y_pred=fit_data.predict(df_test_norm)
acc1 = mean_squared_error(y_test,y_pred)
print(acc1)
acc2 = r2_score(y_test,y_pred)
print(acc2)
plt.figure(figsize=(20,20))
plt.scatter(dates,y_pred,label="Predicted Closing Stock Price")
plt.scatter(dates,y_test,c="#DDA0DD",label="Actual Closing Stock Price")
plt.title("Predicted Closing Price of Stock and Actual Closing Price of Stock using Linear Regression",fontsize=20)
plt.xlabel('Time in Years',fontsize=20)
plt.ylabel('Stock Closing Price Predictions(Linear Regression)',fontsize=20)
plt.legend(loc='best')
plt.show()


# In[25]:


#Test Case 1: The MSE obtained by setting random splitter is 0.456 while the best split gives MSE of 3.46
#Test Case 2: Setting max_depth to 80 gives the MSE as 3.4051 on the other hand, setting the default value also gives MSE 3.4051
print("Decision Tree Regressor")
from sklearn.tree import DecisionTreeRegressor
dtree_regressor = DecisionTreeRegressor(criterion='mse', splitter='best',random_state=1,presort=True)
dtree_regressor.fit(df_train_norm,y_train)
dtree_pred=dtree_regressor.predict(df_test_norm)
acc5 = mean_squared_error(y_test,dtree_pred)
print(acc5)
acc6 = r2_score(y_test,dtree_pred)
print(acc6)
plt.figure(figsize=(20,20))
plt.scatter(dates,dtree_pred,label="Predicted Closing Stock Price")
plt.scatter(dates,y_test,c="#B0E0E6",label="Actual Closing Stock Price")
plt.title("Predicted Closing Price of Stock and Actual Closing Price of Stock using Decision Tree Regressor",fontsize=20)
plt.xlabel('Time in Years',fontsize=20)
plt.ylabel('Stock Closing Price Predictions(Decision Tree Regressor)',fontsize=20)
plt.legend(loc='best')
plt.show()


# In[ ]:




