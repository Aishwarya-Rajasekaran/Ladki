from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
# loading the iris dataset
import pickle
iris = pd.read_csv("cycle-data.csv")
X=iris.to_numpy()

print(X.shape)
x_train=X[:,:-1]
Y=X[:,-1]

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(x_train, Y, test_size=0.2,random_state=0)
print(y_train)
# training a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test[1,:].reshape(1,-1))
print("hettt")
print(gnb_predictions)
# accuracy on X_test
accuracy = gnb.score(X_test, y_test)
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
import numpy as np
inp = [19, 162.59, 52, 34.4, 86]
i = np.asarray(inp)
x = i.reshape(1, -1)
a = pickle_model.predict(x)
print(a[0])
import json
with open("map.json", 'r') as f:
    idx_class = json.load(f)
predicted_label = idx_class[a[0]]
print(predicted_label)
#print(accuracy)
import pickle

#
# Create your model here (same as above)
#

# Save to file in the current working directory
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(gnb, file)
