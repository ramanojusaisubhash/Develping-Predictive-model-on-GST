import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


Xtrain_data = pd.read_csv("E:\\GST-ML_model\\TRAIN\\X_Train_Data_Input.csv")
Ytrain_data = pd.read_csv("E:\\GST-ML_model\\TRAIN\\Y_Train_Data_Target.csv")

Xtest_data = pd.read_csv("E:\\GST-ML_model\\TEST\\X_Test_Data_Input.csv")
Ytest_data = pd.read_csv("E:\\GST-ML_model\\TEST\\Y_Test_Data_Target.csv")

#To find the Nan values in datasets
print("The Columns which contains Nan values in Xtrain :\n",Xtrain_data.columns[Xtrain_data.isna().any()])

print("The Columns which contains Nan values in Xtest :\n",Xtest_data.columns[Xtest_data.isna().any()])

#Replacing the Nan values with mean() of specific column for Xtrain_data
Xtrain_data.Column0 = Xtrain_data.Column0.fillna(Xtrain_data.Column0.mean())
Xtrain_data.Column3 = Xtrain_data.Column3.fillna(Xtrain_data.Column3.mean())
Xtrain_data.Column4 = Xtrain_data.Column4.fillna(Xtrain_data.Column4.mean())
Xtrain_data.Column5 = Xtrain_data.Column5.fillna(Xtrain_data.Column5.mean())
Xtrain_data.Column6 = Xtrain_data.Column6.fillna(Xtrain_data.Column6.mean())
Xtrain_data.Column8 = Xtrain_data.Column8.fillna(Xtrain_data.Column8.mean())
Xtrain_data.Column9 = Xtrain_data.Column9.fillna(Xtrain_data.Column9.mean())
Xtrain_data.Column14 = Xtrain_data.Column14.fillna(Xtrain_data.Column14.mean())
Xtrain_data.Column15 = Xtrain_data.Column15.fillna(Xtrain_data.Column15.mean())

Xtrain_nan_columns = Xtrain_data.columns[Xtrain_data.isna().any()]

print("\n\nThe Columns which contain NaN values in Xtrain After filling with mean():", Xtrain_nan_columns if len(Xtrain_nan_columns) > 0 else None)

#Replacing the Nan values with mean() of specific column for XTest_data 
Xtest_data.Column0 = Xtest_data.Column0.fillna(Xtest_data.Column0.mean())
Xtest_data.Column3 = Xtest_data.Column3.fillna(Xtest_data.Column3.mean())
Xtest_data.Column4 = Xtest_data.Column4.fillna(Xtest_data.Column4.mean())
Xtest_data.Column5 = Xtest_data.Column5.fillna(Xtest_data.Column5.mean())
Xtest_data.Column6 = Xtest_data.Column6.fillna(Xtest_data.Column6.mean())
Xtest_data.Column8 = Xtest_data.Column8.fillna(Xtest_data.Column8.mean())
Xtest_data.Column9 = Xtest_data.Column9.fillna(Xtest_data.Column9.mean())
Xtest_data.Column14 = Xtest_data.Column14.fillna(Xtest_data.Column14.mean())
Xtest_data.Column15 = Xtest_data.Column15.fillna(Xtest_data.Column15.mean())

Xtest_nan_columns = Xtest_data.columns[Xtest_data.isna().any()]

print("\n\nThe Columns which contains Nan values in Xtrain After filling with mean(): ",Xtest_nan_columns if len(Xtest_nan_columns) > 0 else None)



#Converting the Data in the StandardScaler 
s= StandardScaler()

#Removing the ID column from the dataset i.e,.1st column
X_train = Xtrain_data.iloc[:,1:]
X_test = Xtest_data.iloc[:,1:]
y_train_Scaled = Ytrain_data.iloc[:,1:].values.ravel()
y_test_Scaled = Ytest_data.iloc[:,1:].values.ravel()

#Fiting of the Xtraining data to S
s.fit(X_train)

X_train_Scaled = s.transform(X_train)
X_test_Scaled = s.transform(X_test)

print("Standard Scaler Data of Train :\n",X_train_Scaled)
print(X_train_Scaled.shape[0])#printing the no of rows in traing

print("\nStandard Scaler Dara of Test :\n",X_test_Scaled)
print(X_test_Scaled.shape[0])#printing the no of rows in testing


#Navie Bayes Theorem
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = GaussianNB()
model.fit(X_train_Scaled, y_train_Scaled)

y_pred = model.predict(X_test_Scaled)

print("\nNavie Bayes Theorem\n")
#Accuracy 
accuracy = accuracy_score(y_test_Scaled, y_pred)
print("Accuracy : ",accuracy)

#Precision
precision = precision_score(y_test_Scaled, y_pred, average='weighted')  # Use 'binary' for binary classification
print("Precision : ",precision)

#Recall
recall = recall_score(y_test_Scaled, y_pred, average='weighted')
print("Recall : ",recall)

#F1 Score
f1 = f1_score(y_test_Scaled, y_pred, average='weighted')
print("F1 : " , f1)


#Artificial nural network
from sklearn.neural_network import MLPClassifier
ann_model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=100, random_state=42)
ann_model.fit(X_train_Scaled, y_train_Scaled)
y_pred = ann_model.predict(X_test_Scaled)

accuracy = accuracy_score(y_test_Scaled, y_pred)
precision = precision_score(y_test_Scaled, y_pred)
recall = recall_score(y_test_Scaled, y_pred)
f1 = f1_score(y_test_Scaled, y_pred)

print("\nArtificial Nural Network\n")
print("ANN Accuracy: {:f}".format(accuracy))
print("Precision: {:f}".format(precision))
print("Recall: {:f}".format(recall))
print("F1 score: {:f}".format(f1))

#Support vector machine
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
support =svm.LinearSVC(random_state=20)
support.fit(X_train_Scaled, y_train_Scaled)
y_pred = support.predict(X_test_Scaled)

print("\nSupport vector machine\n")
print("Accuracy:", accuracy_score(y_test_Scaled, y_pred))

print("precision",precision_score(y_test_Scaled, y_pred))

print("Recall",recall_score(y_test_Scaled, y_pred))

print("F1 score",f1_score(y_test_Scaled, y_pred))


