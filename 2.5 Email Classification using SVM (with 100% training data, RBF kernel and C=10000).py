""" 
    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
    RBF kernel used in order to improve the accuracy of the algorithm
"""
    
import sys
from time import time
sys.path.append("C:\\Users\\HP\\Desktop\\ML Code\\")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#defining the classifier
clf = SVC(C=10000, kernel='rbf')

#predicting the time of train and testing
t0 = time()
clf.fit(features_train, labels_train)
print("\nTraining time:", round(time()-t0, 3), "s\n")
t1 = time()
pred = clf.predict(features_test)
print("Predicting time:", round(time()-t1, 3), "s\n")

print("Prediction for element 10th, 26th and 50th are:", pred[10], pred[26], pred[50])

#calculating and printing the accuracy of the algorithm
print("Accuracy of SVM Algorithm with 100% Training Data, RBF kernel and C=10000: ", clf.score(features_test, labels_test))

#calculating the number of Chris' emails in a 1700 test dataset
#Chris has label 1
print('Number of events predicted in Chris class is', sum(clf.predict(features_test) ==1))
