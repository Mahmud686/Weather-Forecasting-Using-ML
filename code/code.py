# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:37:18 2020

@author: mahmu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:11:40 2020

@author: mahmu
"""

import pandas as pd 
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


#file

f1=open('DT.txt', "w")
f2=open('AB.txt',"w")
p1 = open('RF.txt', "w")
p2 = open('GB.txt',"w")



df = pd.read_csv("weather.csv")
#drop temperature column from input
X = df.drop('Temperature(°F)', axis='columns')
Y = df['Temperature(°F)']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 1)

#Decision tree ragressor
Classifier1 = DecisionTreeClassifier()
Classifier1.fit(X_train, Y_train)
A_pred = Classifier1.predict(X_test)
print("\nPredicted Values of Decision tree :\n", A_pred)
print('\nMean Absolute Error of Decision tree:',metrics.mean_absolute_error(Y_test,A_pred))



#Adaboost Classifier
Classifier2 = AdaBoostClassifier()
Classifier2.fit(X_train, Y_train)
B_pred = Classifier2.predict(X_test)
print("\nPredicted Values of Adaboost:\n", B_pred)
print('\nMean Absolute Error of Adaboost:',metrics.mean_absolute_error(Y_test,B_pred))



#random forest ragressor
Classifier3 = RandomForestClassifier()
Classifier3.fit(X_train, Y_train)
Y_pred = Classifier3.predict(X_test)
print("\nPredicted Values of Randomforest:\n", Y_pred)
print('\nMean Absolute Error of Randomforest:',metrics.mean_absolute_error(Y_test,Y_pred))


#gradient boosting Classifier
Classifier4 = GradientBoostingClassifier()
Classifier4.fit(X_train, Y_train)
K_pred = Classifier4.predict(X_test)
print("\nPredicted Values of GradientBoostingClassifier:\n", K_pred)
print('\nMean Absolute Error of GradientBoostingClassifier:',metrics.mean_absolute_error(Y_test,K_pred))






print("\n")
for k in A_pred:
    print(k, end="\n", file=f1)
f1.close

with open('DT.txt') as infile, open('DT101.csv','w') as outfile:
    for line in infile:
        outfile.write(line)
        




print("\n")
for l in B_pred:
    print(l, end="\n", file=f2)
f2.close

with open('AB.txt') as infile, open('AB101.csv','w') as outfile:
    for line in infile:
        outfile.write(line)

       

print("\n")       
for i in Y_pred:
            
        print(i, end="\n", file=p1)
p1.close

with open('RF.txt') as infile, open('RF101.csv','w') as outfile: 
 for line in infile: 
            
  outfile.write(line)
  
  
  
  
  
print("\n")       
for j in K_pred:
            
        print(j, end="\n", file=p2)
p2.close

with open('GB.txt') as infile, open('GB101.csv','w') as outfile: 
 for line in infile: 
            
  outfile.write(line)

accuracy_test = accuracy_score(Y_test,A_pred)
print('accuracy_score of DT : ', accuracy_test*100)
accuracy_test = accuracy_score(Y_test,B_pred)
print('accuracy_score on AB : ', accuracy_test*100)
accuracy_test = accuracy_score(Y_test,Y_pred)
print('accuracy_score on RF : ', accuracy_test*100)
accuracy_test = accuracy_score(Y_test,K_pred)
print('accuracy_score on GB : ', accuracy_test*100)


ad1 = []
for k in range(len(A_pred)):
   ad1.append(k)
   
ad2 = []
for l in range(len(B_pred)):
   ad2.append(l)

ad3 = []
for i in range(len(Y_pred)):
   ad3.append(i)
   
ad4 = []
for j in range(len(K_pred)):
   ad4.append(j)  
   
plt.figure()
plt.plot(ad1, Y_test, color="green", label="ACTUAL", linewidth=1)
plt.plot(ad1, A_pred, color="yellow", label="Decision Tree Classifier", linewidth=1)
plt.plot(ad2, B_pred, color="cyan", label="AdaBoost Classifier", linewidth=1)
plt.plot(ad3, Y_pred, color="blue", label="Random Forest Classifier", linewidth=1)
plt.plot(ad4, K_pred, color="red", label="Gradient Boosting Classifier", linewidth=1)
plt.xlabel("Index")
plt.ylabel("target")
plt.title("RandomForest vs GradientBoosting")
plt.legend()
plt.show()  
