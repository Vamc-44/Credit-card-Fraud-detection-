#Importing Libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Pandas options
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None, 'display.max_columns', None)

#Plotting options
%matplotlib inline
mpl.style.use('ggplot')
sns.set(style='whitegrid')

#Reading data
transactions = pd.read_csv('creditcard.csv')

#Data Cleansing
transactions.shape

transactions.info()

transactions.isnull().any().any()

transactions.sample(7)

transactions['Class'].value_counts()

transactions['Class'].value_counts(normalize=True)

