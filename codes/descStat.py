'''
Created on 13 mai 2016

@author: Gen Yang
'''
import numpy as np
import matplotlib.pyplot as plt
import manipData as md# custom class for data manipulation
#import sklearn


#===============================================================================
# Local config & variables
#===============================================================================

file_meta = 'census_income_metadata.txt'
file_learn = 'census_income_learn.csv'
file_test = 'census_income_test.csv'

plt.rcParams["axes.titlesize"] = 10


#===============================================================================
# Script pour les statistiques descriptives
#===============================================================================


# Import learning data 
data = md.manipData(file_meta=file_meta,file_data=file_learn)
 

# Quick check & summary of the dataframe
data.df.info()


# Take out the 'instance weight' attribute for the learning data
data.drop_col(name = 'instance weight') 


# Generate the output summary tables for numeric attributes
temp = data.df.describe()
miss = (1- temp.loc['count',:]/ data.df.shape[0]) *100
miss.name ='Miss(%)' # Percent of missing data
temp=temp.append(miss).transpose()
temp[['count','min','25%','50%','75%','max']] = temp[['count','min','25%','50%','75%','max']].astype(int)
temp['Miss(%)'] = temp['Miss(%)'].astype(float)
temp.to_csv('Descriptive_statistics_numeric.csv',float_format='%0.2f')


# Generate the output summary tables for categorical attributes
temp = data.df.describe(include=['category'])
miss = (1- temp.loc['count',:]/ data.df.shape[0]) *100
miss.name ='Miss(%)' # Percent of missing data
miss = miss.astype(float)
temp=temp.append(miss).transpose()
temp['count'] = temp['count'].astype(int)
temp['Miss(%)'] = temp['Miss(%)'].astype(float)
temp.to_csv('Descriptive_statistics_categorical.csv',float_format='%0.2f')


# Generate the output bar plot for numeric attributes
fig = plt.figure()
data.df.select_dtypes(include=[np.number]).hist(xlabelsize=6, ylabelsize=6)
plt.savefig('Histogram of continuous attributes.png',dpi=800, bbox_inches='tight') 
#plt.show()
plt.close(fig)


# Generate the output bar plot for categorical attributes
for col in data.df.select_dtypes(include=['category']):
    fig = plt.figure()
    data.df[col].value_counts().plot(kind='barh')
    plt.savefig(col + '_histo.png',dpi=300, bbox_inches='tight')
    plt.close(fig) 



