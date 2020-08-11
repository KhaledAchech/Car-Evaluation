"""
this programme has for purpose to use the KNN algorithme
to classify cars
"""

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

#reading our data
data = pd.read_csv("Data/car.data")
print(data.head())

#converting the attributs into numeric values for easier use
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

#x : features and y : labels
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

print(x_train, y_test)

"""
Defining the model in this case am using KNN with 9(you can play arround with the number to get a better accuracy) 
wich is the number of neighbors to check.
note that n_neighbors always should be an odd number
"""
model = KNeighborsClassifier(n_neighbors = 9)

#training
model.fit(x_train, y_train)

#finding the accuracy
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
#this is to make it more presentable becuase all of the attributes have numeric values
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("predicted : ", names[predicted[x]], "data : ", x_test[x], "Actual : ", names[y_test[x]])


