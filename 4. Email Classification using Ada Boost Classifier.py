""" 
    Using an Ada Boost Classifier to identify emails by their authors
    authors and labels:
    Sara has label 0
    Chris has label 1
"""  

import sys
from time import time
sys.path.append("C:\\Users\\HP\\Desktop\\ML Code\\")
from email_preprocess import preprocess
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# defining the classifier
clf = AdaBoostClassifier(n_estimators=100, random_state=0)

#predicting the time of train and testing
t0 = time()
clf.fit(features_train, labels_train)
print("\nTraining time:", round(time()-t0, 3), "s\n")
t1 = time()
pred = clf.predict(features_test)
print("Predicting time:", round(time()-t1, 3), "s\n")

#calculating and printing the accuracy of the algorithm
print("Accuracy of Ada Boost Classifier: ", accuracy_score(pred,labels_test))