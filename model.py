import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import joblib

#this imports the necessary libraries we need for machine learning 

data = pd.read_csv("myopia.csv", sep=";", skipinitialspace=True)

data.columns = data.columns.astype(str)
data.columns = data.columns.str.strip()
#reading the file of data, and removing extra spaces

predict = (data['MYOPIC'])

data = data[["SPHEQ", "AL", "ACD"]];
#separating data into two sets: predict is our final answer of whether the subject is diagnosed with myopia
#the data dataset contains the factors
X = np.array(data)  
y = np.array(predict)  
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)
#we split our data into arrays, X is our factors, y is our classification. 
#we create two sets, one set to train our model and another to test our data, to determine
#if it is accurate. we split our testing data set into 20 percent of the total data set.
classifier = LogisticRegression()
classifier.fit(x_train, y_train)
#we fit our model into the training data set to train it
acc = classifier.score(x_test, y_test)
#acc is the testing data set
#our accuracy means how accurate the model is in identifying whether the subject will be myopic.
print(acc)
joblib.dump(classifier, 'classifier.pkl')
# this saves our model so we can access it through our application

print("succesfully saved")
#this tells us that the model was succesfully saved. 

