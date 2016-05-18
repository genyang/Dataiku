'''
Created on 17 mai 2016

@author: Gen Yang
'''
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

class manipData(object):
    '''
    Custom class for data storage and manipulation
    
    :param df: the dataframe we work on
    :type df: pandas.DataFrame 
    '''


    def __init__(self, file_meta, file_data, categories=None,nrows=None):
        '''
        Import data
        :param file_meta: meta-data file
        :param file_data: the data file
        :param nrows: the number of data instances to be used (only for test purpose)
        
        :param categories: if specified, it will be used to help set up categorical
                variables. The format of this input is :
                dictionary of {<column name> : list of categories}
        '''
        
        # Creation of the list of the columns' names from 'metadata'

        names_attrbiuts = pd.read_csv(file_meta, sep=':', skipinitialspace=True,
                                      skiprows =142, usecols=[0],comment='|',dtype='string',header=None)
        
        col_names = names_attrbiuts[0].values.tolist() 
        col_names.append('class') # we name the objective variable "class"
        
        
        # Find the data type of each column
        
        col_types = pd.read_csv(file_meta, sep=')', skipinitialspace=True, skiprows =81, nrows=40, dtype='string',header=None)
        
        col_types.iloc[:,0] = col_types.iloc[:,0].apply(lambda x : x.split('(')[-1])
        col_types.iloc[:,1] = col_types.iloc[:,1].apply(lambda x : np.int32 if x == 'continuous' else object)
        
        dict_types = col_types.set_index(0)[1].to_dict()
        dict_types['instance weight'] = np.float32
        dict_types['class'] = object
        
         
        # Values to be considered as NA/NaN
        na_val = ['?']
        
        # raw dataframe
        data_raw = pd.read_csv(file_data, sep=',', skipinitialspace=True, nrows = nrows,
                      names = col_names, na_values=na_val, dtype = dict_types)
        
        # Convert columns containing 'string object' to 'categorical variable'
        for col in data_raw.columns:
            if data_raw[col].dtype == 'object':
                if categories is None:
                    data_raw[col] = data_raw[col].astype('category')
                else : # if a category dictionary is given as parameter
                    if col not in categories:
                        raise Exception('Incomplete category dictionary lacking:',col)
                    data_raw[col] = data_raw[col].astype('category').cat.set_categories(categories[col])
        
        self.df = data_raw
        
        
    def get_categories(self):
        '''
        Create a dictionary for all categorical variables of format:
            dictionary of {<column name> : list of categories}
        ''' 
        dictionary = {}
        for col in self.df.columns:
            if str(self.df[col].dtype) == 'category':
                dictionary[col] = self.df[col].cat.categories
        return dictionary
    
    def drop_col(self,name):
        '''
        Drop a column of the dataframe : transform the initial dataframe so that
        the new one does not contain the droped column.
        The droped column is returned by the method.
        '''
        droped = self.df[name].copy()
        self.df = self.df.drop(name, axis=1)
        return droped
        
    def clean_na(self,method):
        '''
        Clean the missing data
        '''
        
        # This method consists in just deleting any instance containing missing data
        if method == 'full amputation':
            self.df.dropna(inplace=True)
        elif method == 'completion': #custum procedure to complete data
            
            # For the attribute 'state of previous residence', complete by top mode 
            self.df['state of previous residence'].fillna(value = 'Not in universe',inplace=True)
            
            # For 'migration code' and 'hispanic origin'
            if 'NA' not in self.df['hispanic origin'].cat.categories: # keep coherence with metadata
                self.df['hispanic origin'] = self.df['hispanic origin'].cat.add_categories(['NA']) 
            self.df['hispanic origin'].fillna(value = 'NA',inplace=True)
            self.df['migration code-change in msa'].fillna(value = 'Not in universe',inplace=True)
            self.df['migration code-change in reg'].fillna(value = 'Not in universe',inplace=True)
            self.df['migration code-move within reg'].fillna(value = 'Not in universe',inplace=True)
            self.df['migration prev res in sunbelt'].fillna(value = 'Not in universe',inplace=True)
            
            # For 'country of birth'
            if 'NA' not in self.df['country of birth father'].cat.categories: 
            # As we always modify these attributs together, one test is sufficient 
                self.df['country of birth father'] = self.df['country of birth father'].cat.add_categories(['NA'])
                self.df['country of birth mother'] = self.df['country of birth mother'].cat.add_categories(['NA'])
                self.df['country of birth self'] = self.df['country of birth self'].cat.add_categories(['NA'])
            
            self.df['country of birth father'].fillna(value = 'NA',inplace=True)
            self.df['country of birth mother'].fillna(value = 'NA',inplace=True)
            self.df['country of birth self'].fillna(value = 'NA',inplace=True)
        elif method == 'half amputation':
            # Only complete 'migration code' and 'hispanic origin'
            if 'NA' not in self.df['hispanic origin'].cat.categories:
                self.df['hispanic origin'] = self.df['hispanic origin'].cat.add_categories(['NA'])
            self.df['hispanic origin'].fillna(value = 'NA',inplace=True)
            self.df['migration code-change in msa'].fillna(value = 'Not in universe',inplace=True)
            self.df['migration code-change in reg'].fillna(value = 'Not in universe',inplace=True)
            self.df['migration code-move within reg'].fillna(value = 'Not in universe',inplace=True)
            # amputate the rest
            self.df.dropna(inplace=True)
        else:
            raise Exception('Unspecified cleaning method:',method)
        
    def clean_na_generic(self):
        '''
        A more generic method for missing values handling, used for the test data file 
        '''
        # storage of replacement mapping of missing values per attribute 
        dictionary = {}
        
        for col in self.df.columns:
            if str(self.df[col].dtype) == 'category':
                cat = self.df[col].cat.categories
                if 'Not in universe' in cat:
                    dictionary[col] = 'Not in universe'
                elif 'NA' in cat :
                    dictionary[col] = 'NA'
                else : # use the most frequent mode to fill NaN
                    dictionary[col] = self.df[col].mode()[0]
            else: #numeric attribute, we use mean
                dictionary[col] = int(self.df[col].mean())
        
        self.df = self.df.fillna(value= dictionary)
        
        
    
    def processData(self,method='Binarize', vect=None):
        '''
        Preprocess the data for application with sci-kit learn's 
        :param X: the numpy.ndarray of features attributs
        :param Y: the vector of class values
        :param vect: the vector mapping for the binary encoding, leave 'None' for training
        '''
        if method == 'Binarize':
            
            #===================================================================
            # train = self.df.drop('class',axis=1)
            # X_dict = [dict(r.iteritems()) for _, r in train.iterrows()]
            # X = DictVectorizer(sparse=False).fit_transform(X_dict)
            #===================================================================
            
            Y = np.array(self.df['class'].apply(lambda x: -1 if x=='- 50000.' else 1))
            
            X_dict = self.df.drop('class',axis=1).T.to_dict().values()           
            if vect is None: #The transformation matrix is not yet created
                vect = DictVectorizer(sparse=False)
                X = vect.fit_transform(X_dict)
                return X,Y,vect
            else:
                X = vect.transform(X_dict)
                return X,Y
        else :
            raise Exception('Not implemented yet:',method)
        