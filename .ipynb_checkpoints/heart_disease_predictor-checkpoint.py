#!/usr/bin/env python
# coding: utf-8

# #### Heart Disease Predictor

# In[4]:


#Import the Libraries
import numpy as np # To work with arrays
import pandas as pd #To work with the data
#Importing Sk learn libraries/ Modules
from sklearn.model_selection import train_test_split # Module to split the data into training and testing
from sklearn.linear_model import LogisticRegression # To build the module
from sklearn.metrics import accuracy_score # Evaluation
from sklearn.preprocessing import StandardScaler


# In[5]:


#Importing the data
df = pd.read_csv("Data/heart_disease_data.csv")


# In[6]:


#First 5 records
df.head()


# In[10]:


#Last 5 records
df.tail()


# In[12]:


#Getting the dimensions of our data
df.shape


# In[14]:


#Get a summary of the DataFrame (index, columns, data types, memory usage)
df.info()


# In[16]:


#Missing values
df.isnull().sum()


# In[18]:


#Descriptive statistics
df.describe()


# In[20]:


# Count unique values in the 'target' column
df['target'].value_counts()


# -1 $\rightarrow$ Heart Disease
# -0 $\rightarrow$ Healthy Patient

# #### Splittin Features and Target

# In[24]:


# Create a new DataFrame X containing all columns except 'target'
X = df.drop(columns='target',axis=1)
X


# In[26]:


# Select the 'target' column from the DataFrame
Y = df['target']
Y


# #### Training and Testing

# In[29]:


# Split the data into training and testing sets
# X_train and Y_train contain the training data (features and target)
# X_test and Y_test contain the test data (features and target)
# test_size=0.2 specifies that 20% of the data will be used for testing
# stratify=Y ensures that the class distribution is the same in both training and test sets
# random_state=45 ensures reproducibility of the split

X_train , X_test , Y_train , Y_test = train_test_split( X, Y ,test_size = 0.2, stratify= Y, random_state= 45)


# In[31]:


# Initialize Standard Scalar model
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[33]:


# Display the number of rows and columns in the feature matrix X
X.shape


# In[35]:


# Display the number of rows and columns in the training feature set X_train
X_train.shape


# In[37]:


# Display the number of rows and columns in the testing feature set X_test
X_test.shape


# In[39]:


# Display the number of rows and columns (if any) in the training target set Y_train
Y_train.shape


# In[41]:


# Display the number of rows and columns (if any) in the testing target set Y_test
Y_test.shape


# #### Model Building
# 
# LogisticRegression is used to create the model at the end of the day; it is s binary classification

# In[44]:


# Initialize Logistic Regression model

model = LogisticRegression(max_iter=500)


# In[46]:


#Train the model

model.fit(X_train,Y_train)


# #### Evaluation

# In[49]:


#Accuracy score on training data

X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction,Y_train)
print(f"The training accuracy is {training_accuracy}")


# In[51]:


#Accuracy score on testing data

X_test_prediction = model.predict(X_test)
testing_accuracy = accuracy_score(X_test_prediction,Y_test)
print(f"The testing accuracy is {testing_accuracy}")


# Building the Predicting System:

# In[59]:


#Tuple of the feature data ---> Which is my input
input_data = (37,1,2,130,250,0,1,187,0,3.5,0,0,2)

#Convertin it into numpy array
input_data = np.asarray(input_data)

#Reshaping into required intput form
input_data = input_data.reshape(1,-1)

#Predict the output for the given input data using the trained model
prediction = model.predict(input_data)

prediction

if prediction[0] == 0:
    print("✅ Good News the patient doesn't have any heart disease")
else:
    print("⚠️ The Patient should visit the doctor")


# In[ ]:




