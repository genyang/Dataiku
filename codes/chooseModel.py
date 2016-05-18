'''
Created on May 17, 2016

@author: Gen Yang
'''
import numpy as np
import time
import manipData as md
from sklearn import tree,linear_model
from sklearn.cross_validation import KFold

file_meta = 'census_income_metadata.txt'
file_learn = 'census_income_learn.csv'
file_test = 'census_income_test.csv'



#===============================================================================
# Script pour le choix du modele
#===============================================================================


# Import learning data 
data = md.manipData(file_meta=file_meta,file_data=file_learn,nrows=None)
weights = data.drop_col(name = 'instance weight') 
data.clean_na('completion')

X,Y = data.processData()


'''
Perform a 10-fold cross validation to select among the 2 tested algorithms :
    - Decision Tree
    - Logistic Regression
''' 
len_total = len(Y)
kf = KFold(len_total,n_folds=10)
classif_dt = tree.DecisionTreeClassifier()
classif_lg = linear_model.LogisticRegression(solver='lbfgs')

score_dt = 0.
score_lg = 0.
time_dt = 0.
time_lg = 0.
time_dt_ini = 0.
time_dt_end = 0.
time_lg_ini = 0.
time_lg_end = 0.

for train_id, val_id in kf:
    
    # Decision Tree
    
    time_dt_ini = time.clock() 
    classif_dt = classif_dt.fit(X[train_id,:], Y[train_id],sample_weight=np.array(weights[train_id]))
    pred_dt = classif_dt.predict(X[val_id,:])
    time_dt_end = time.clock() 
    
    # Logistic Regression
    
    time_lg_ini = time.clock() 
    classif_lg = classif_lg.fit(X[train_id,:], Y[train_id],sample_weight=np.array(weights[train_id]))
    pred_lg = classif_lg.predict(X[val_id,:])
    time_lg_end = time.clock() 
    
    time_dt = time_dt + time_dt_end - time_dt_ini
    time_lg = time_lg + time_lg_end - time_lg_ini

    score_dt += np.sum(pred_dt == Y[val_id])
    score_lg += np.sum(pred_lg == Y[val_id])

print "score of Decision Tree: " + str(round(score_dt/len_total,2))
print "time : ", time_dt
print "score of Logistic Regression: " + str(round(score_lg/len_total,2))
print "time : ", time_lg