import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, make_scorer, accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc

from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
import matplotlib.pyplot as plt

scoring = {'AUC': make_scorer( roc_auc_score, average='weighted', multi_class = 'ovr', needs_proba=True), 
            'Accuracy': make_scorer(accuracy_score)}

## Read data----
X = pd.read_table( "Data/OUP_Inp.txt", sep = " ", header = None )
Map = pd.read_table( "Data/Labels.map", sep = " ", header = None )
y = Map[2] - 1

# split into train and test sets----
X_train, X_test, y_train, y_test, Net_train, Net_test, Cls_train, Cls_test = train_test_split( X, y, Map[1], Map[3], stratify=y, 
                                    train_size = 0.8,  random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# set parameters----
# 
clf = OneVsRestClassifier( MLPClassifier( max_iter=1000, random_state=0
         , hidden_layer_sizes = [ 100, 160, 160, 60 ]
         , alpha = 0.05, activation = 'relu', solver = 'adam'
      ), n_jobs = -1 )

scores = cross_val_score( clf, X_train, y_train, cv = 5, n_jobs = -1)

clf.fit( X_train, y_train )

print("train set score: {:.2f}".format( clf.score(X_train, y_train) ))
print("Test set score: {:.2f}".format( clf.score(X_test, y_test) ))
print("Mean CV score: {:.2f}".format( scores.mean() ))

#y_score = clf.decision_function( X_test )
y_score = clf.predict_proba( X_test )
y_pred = clf.predict( X_test )

cm = confusion_matrix( y_test, y_pred, normalize = 'true' )
cm = cm.round(3)
pd.DataFrame( cm ).to_csv( "conf_mat_0.8_nm_ovr2.txt", sep = " ", header = False, index = False )
