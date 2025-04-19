# heart_disease_predictor.py



#Import the Libraries
import numpy as np # To work with arrays
import pandas as pd #To work with the data
#Importing Sk learn libraries/ Modules
from sklearn.model_selection import train_test_split # Module to split the data into training and testing
from sklearn.linear_model import LogisticRegression # To build the module
from sklearn.metrics import accuracy_score # Evaluation
from sklearn.preprocessing import StandardScaler



#Importing the data
df = pd.read_csv("Data/heart_disease_data.csv")



# Prepare features and target
X = df.drop(columns='target', axis=1)


# Select the 'target' column from the DataFrame
Y = df['target']



# #### Training and Testing



# Split the data into training and testing sets
# X_train and Y_train contain the training data (features and target)
# X_test and Y_test contain the test data (features and target)
# test_size=0.2 specifies that 20% of the data will be used for testing
# stratify=Y ensures that the class distribution is the same in both training and test sets
# random_state=45 ensures reproducibility of the split

X_train , X_test , Y_train , Y_test = train_test_split( X, Y ,test_size = 0.2, stratify= Y, random_state= 45)



# Initialize Standard Scalar model
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Initialize Logistic Regression model

model = LogisticRegression(max_iter=500)


#Train the model

model.fit(X_train,Y_train)


#### Evaluation


#Accuracy score on training data

X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction,Y_train)
print(f"The training accuracy is {training_accuracy}")



#Accuracy score on testing data

X_test_prediction = model.predict(X_test)
testing_accuracy = accuracy_score(X_test_prediction,Y_test)
print(f"The testing accuracy is {testing_accuracy}")




# Building the Predicting System:

def predict_heart_disease(input_data):
    #Tuple of the feature data ---> Which is my input
    input_data = (37,1,2,130,250,0,1,187,0,3.5,0,0,2)

    #Convertin it into numpy array
    input_data = np.asarray(input_data)
    
    #Reshaping into required intput form
    input_data = input_data.reshape(1,-1)
    
    #Predict the output for the given input data using the trained model
    prediction = model.predict(input_data)
    
    
    if prediction[0] == 0:
        print("✅ Good News the patient doesn't have any heart disease")
    else:
        print("⚠️ The Patient should visit the doctor")




