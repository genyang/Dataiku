'''
Created on May 17, 2016

@author: Gen Yang
'''

import manipData as md
import numpy as np
from sklearn.linear_model import LogisticRegression


file_meta = 'census_income_metadata.txt'
file_learn = 'census_income_learn.csv'
file_test = 'census_income_test.csv'



#===============================================================================
# Script for the interpretation of most impactful factors with Logistic Regression 
#===============================================================================


# Import learning data 
data_learn = md.manipData(file_meta=file_meta,file_data=file_learn,nrows=100)
weights = data_learn.drop_col(name = 'instance weight') 
data_learn.clean_na('completion')

cat = data_learn.get_categories()

X_learn,Y_learn,vect_learn = data_learn.processData()

# Logistic Regression

classif_lg = LogisticRegression(solver='lbfgs')
classif_lg = classif_lg.fit(X_learn, Y_learn,sample_weight=np.array(weights))

data_test = md.manipData(file_meta=file_meta,file_data=file_test,categories=cat)
weights = data_test.drop_col(name = 'instance weight') 
data_test.clean_na('completion')
data_test.clean_na_generic()

#print "Number of non treated missing data : ", data_test.df.isnull().sum()

data_test.df = data_test.df.dropna() # We delete all rows with missing data at this stage
 
X_test,Y_test = data_test.processData(vect=vect_learn)
pred = classif_lg.predict(X_test)

print 'Predictive performance (%): ' + str(round(float(np.sum(pred == Y_test))/len(Y_test)*100,2))

# Determine the top N most impactful attributes on the prediction task
 

sorted_id = classif_lg.coef_[0].argsort()

print "top N features (values) :"
N = 20
for i in range(1,N+1):
    print '======= ' + str(i) + ' ======='
    print 'name (values): ' + vect_learn.feature_names_[sorted_id[-i]]
    print 'weight: ' + str(classif_lg.coef_[0,sorted_id[-i]])
    
    
