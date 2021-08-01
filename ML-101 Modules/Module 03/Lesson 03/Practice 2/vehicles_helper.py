import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')


def attributes_counts(dataset):
    """ Basic descriptions of the target attribute.
    
    This module displays 'Class' attribute value counts 
    and visualisation plot of the dataset.
     
    Parameters:
        dataset.
    """
    print("'Class' Value Counts: "+" \n", dataset['Class'].value_counts())
    print("\n Visualisation plot: "+" \n", dataset['Class'].value_counts().plot(x = dataset['Class'], kind='bar'))
   


def all_attrubutes_vizual(dataset, one, two, three):
    """ Visual presentation of all attributes in the dataset.
    
   This module shows all attributes divided into 3 parts
   for better visualization.
   
    Parameters:
        dataset, 
        one: selected attributes for part one
        two: selected attributes for part two 
        three: selected attributes for part three
    """
    print("Part one:"+"\n")
    df1 = dataset[one] 
    sns.pairplot(df1, kind="scatter",  hue="Class", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
    plt.show()
    print("\n")
    print("Part two:"+"\n")
    df2 = dataset[two] 
    sns.pairplot(df2, kind="scatter",  hue="Class", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
    plt.show()
    print("\n")
    print("Part three:"+"\n")
    df3 = dataset[three] 
    sns.pairplot(df3, kind="scatter",  hue="Class", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
    plt.show()
    
   
def corr_plot_list(dataset): 
    """ This module presents correlation plot and list of each attribute"""
    
    print("'Correlation list of each attribute: ")
    corr = dataset.corr()
    corr_abs = corr.abs()
    num_cols = len(dataset)
    num_corr = corr_abs.nlargest(num_cols, 'Class')['Class']
    print(num_corr)
    print("\n")
    print("'Correlation plot of each attribute: "+"\n", dataset.corr()['Class'].sort_values().plot(kind='bar', figsize=(18, 6)))


