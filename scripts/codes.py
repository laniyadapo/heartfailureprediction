#Analysis packages
import pandas as pd
import sklearn as sk
import numpy as np
import scipy.stats as sp

#Visualization packages
import matplotlib.pyplot as plt
import matplotlib as matplot
from matplotlib.ticker import MaxNLocator
import seaborn as sns

#Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

#Models & Sklearn packages
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score as cvs

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from tqdm import tqdm
import datetime

import warnings
warnings.filterwarnings('ignore')


class data_clean():
    def __init__(self, data):
        self.data = data
    
        
    def deal_null_values(self, cols_cat, value):
        '''Replace null values with None or 0 as applicable'''
        self.value = value
        for col in cols_cat:
            self.data[col].replace(np.nan, value, inplace=True)
            
    def deal_null_values_freq(self, cols_cat):
        '''Replace Null values with mode of variable'''
        for col in cols_cat:
            mode_fill = self.data[col].mode()[0]
            self.data[col].replace(np.nan, mode_fill, inplace=True)
    
    def target_visual(self, target):
        '''Visual inspection of target data'''
        self.target = target
        value = (self.data / 1000)
        plt.figure(figsize = (14, 6))
        plt.subplot(1,2,1)
        sns.boxplot(value)
        plt.xlabel(self.target + "(X 1000)")
        plt.title('Box Plot')
        plt.subplot(1,2,2)
        sns.distplot(value, bins=20)
        plt.xlabel(self.target + "(X 1000)")
        plt.title('Distribution Plot')
        plt.show()
    
    def num_features_outliers(self):
        '''Find outliers in numerical features using bivariate analysis'''
        num_features = self.data.select_dtypes(exclude='object').drop('DEATH_EVENT', axis=1).copy()
        f = plt.figure(figsize=(12,20))
        for i in range(len(num_features.columns)):
            f.add_subplot(10, 4, i+1)
            sns.scatterplot(num_features.iloc[:,i], self.data.DEATH_EVENT)
        plt.tight_layout()
        plt.show()
    
    
            
            
class eda_process():
    def __init__(self, data):
        self.data = data
    
    def plot_comp(self, cols):
        "Plot the variables against the target variable"
        for col in cols:
            Score_df = pd.crosstab(self.data[col], self.data['DEATH_EVENT'])
            Score_df.div(Score_df.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
        
    def plot_composite(self, cols):
        "Plot the variables against the target variable"
        for col in cols:
            plt.figure(figsize = (10, 6))
            plt.subplot(1, 2, 1)
            self.data[col].hist()
            plt.subplot(1, 2, 2)
            Score_df = pd.crosstab(self.data[col], self.data['DEATH_EVENT'])
            Score_df.div(Score_df.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
            plt.show()
    
    def plot_composites(self, cols):
        "Plot the variables against the target variable"
        for col in cols:
            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(15, 3)
            self.data[col].hist(ax=ax[0])
            Score_df = pd.crosstab(self.data[col], self.data['DEATH_EVENT'])
            Score_df.div(Score_df.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax=ax[1])
            ax[0].set_xlabel(col, fontsize = 10);
            ax[0].set_ylabel('Counts', fontsize = 10);
            ax[1].set_xlabel(col, fontsize = 10);
            ax[1].set_ylabel('Ratio', fontsize = 10);
            fig.suptitle(col  + ' vs death event relationship', fontsize=12)
            fig.subplots_adjust(top=0.9)
            fig.show()
            
    def groupplot_comp(self, columns):
        "Plot the variables against the target variables using groups" 
        for column in columns:
            info = self.data[column].max()
            bins= [0, (info * 0.25), (info * 0.5), (info * 0.75), info] 
            group= ['Low', 'Average', 'High', 'Very High']
            self.data['A'] = pd.cut(self.data[column], bins, labels=group)
            A= pd.crosstab(self.data['A'], self.data['DEATH_EVENT'])
            A.div(A.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
            
            
    
    def groupplot_compIQR(self, columns):
        "Plot the variables against the target variables using groups" 
        for column in columns:
            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(17, 5)
            self.data[column].hist(ax=ax[0])
            info = self.data[column].describe()
            min = self.data[column].min()
            max = self.data[column].max()
            bins= [min, info['25%'], info['50%'], info['75%'], max] 
            group= ['Low', 'Average', 'High', 'Very High']
            self.data['A'] = pd.cut(self.data[column], bins, labels=group)
            A= pd.crosstab(self.data['A'], self.data['DEATH_EVENT'])
            A.div(A.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax=ax[1])
            ax[0].set_xlabel(column, fontsize = 15);
            ax[0].set_ylabel('Counts', fontsize = 15);
            ax[1].set_xlabel(column, fontsize = 15);
            ax[1].set_ylabel('Ratio', fontsize = 15);
            ax[0].tick_params(axis = 'x', labelsize = 15)
            ax[1].tick_params(axis = 'x', labelsize = 15, rotation = 45)
            ax[0].tick_params(axis = 'y', labelsize = 15)
            ax[1].tick_params(axis = 'y', labelsize = 15)
            #ax[1].legend(loc=(1.02,0), borderaxespad=0, fontsize = 10)
            fig.suptitle(column  + ' vs death event relationship', fontsize=15)
            fig.subplots_adjust(top=0.15)
            
            fig.tight_layout()
            fig.savefig('images/groupIQR.jpg', dpi = 300)
            
    def heatmap_num(self, datatype):
        '''Plot heatmap for numerical features correlation'''
        self.datatype = datatype
        sns.set(font_scale=1.1)
        correlation_train = self.data.corr()
        mask = np.triu(correlation_train.corr())
        plt.figure(figsize=(18, 20))
        plt.title('Correlation of ' + self.datatype, size=20)
        sns.heatmap(correlation_train,
                    annot=True,
                    fmt='.1f',
                    cmap='coolwarm',
                    square=True,
                    mask=mask,
                    linewidths=1,
                    cbar=False)
        plt.show()
    
    def normalize(self, cols):
        self.cols= cols
        for col in self.cols:
            self.data[col +'norm'] = preprocessing.Normalizer(norm='max').transform([self.data[col]])[0]
            
    def data_drop(self, train, cols):
        self.train= train
        self.cols= cols
        for df in [self.train]:
            df.drop(columns = df[self.cols], inplace = True)
    
    def cat_features_plot(self, cols):
        '''Plot to review relationship between independent variables and target'''
        self.cols = cols

        for col in cols:
            plt.figure(figsize = (20, 6))
            plt.subplot(1, 2, 1)
            frequency = self.data[col].value_counts()
            sub_categories = pd.unique(self.data[col])
            sns.barplot(x = sub_categories, y = frequency, data = self.data)
            plt.xticks(rotation = 45)
            plt.xlabel(col)
            plt.ylabel('counts')
            plt.subplot(1, 2, 2)
            sorted= self.data.groupby([col])['DEATH_EVENT'].median().sort_values(ascending=False)
            sns.boxplot(x = self.data[col], y = 'DEATH_EVENT', data = self.data, order=sorted.index)
            plt.xticks(rotation = 45)
            plt.ylabel('DEATH_EVENT')
            plt.tight_layout
            plt.show()

    
    def heatmap(self, featuretype):
        '''Plot heatmap for categorical features'''
        self.featuretype = featuretype
        sns.set(font_scale=1.1)
        self.data = self.data.astype(float)
        correlation_train = self.data.corr()
        mask = np.triu(correlation_train.corr())
        plt.figure(figsize=(18, 20))
        plt.title('Correlation of ' + self.featuretype, size=20)
        sns.heatmap(correlation_train,
                    annot=True,
                    fmt='.1f',
                    cmap='coolwarm',
                    square=True,
                    mask=mask,
                    linewidths=1,
                    cbar=False)
        plt.show()
     
    def num_features_distplot(self, datadesc):
        '''Inspect all numerical feature for skewness'''
        self.datadesc = datadesc
        num_features = self.data.select_dtypes(exclude='object').copy()
        fig = plt.figure(figsize=(12,18))
        for i in range(len(num_features.columns)):
            fig.add_subplot(8,4,i+1)
            sns.distplot(num_features.iloc[:,i].dropna())
            plt.xlabel(num_features.columns[i])
        plt.tight_layout()
        fig.suptitle('Distribution Plot for ' + self.datadesc + ' Features', fontsize=15)
        fig.subplots_adjust(top=0.96)
        plt.show()
        
       
        
   
    def dist_plot(self, titles):
        '''Plotting the target variable to inspect for skewness'''
        self.titles = titles
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(15, 5)
        sns.distplot(self.data, ax=ax[0])
        sns.distplot((np.log(self.data.round(decimals=2))), ax=ax[1])
        ax[0].title.set_text('Distribution of '+ self.titles)
        ax[1].title.set_text('Distribution of Log-Transformed ' + self.titles)
        fig.show()
        print(self.titles + ' has a skew of ' + str(self.data.skew().round(decimals=2)) + 
        ' while the log-transformed ' + self.titles + ' improves the skew to ' + 
        str(np.log(self.data).skew().round(decimals=2)))
    
    def multivarplot(self, data1, data2, data3):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        plt.figure(figsize=(40,15))
        plt.title('Plot of Neighborhood and MSZoning vs SalePrice', size=20)
        sns.barplot(self.data[self.data1], self.data[self.data2], self.data[self.data3], self.data)
        plt.xticks(rotation = 90, fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel('Neighborhood', fontsize=25)
        plt.ylabel('SalePrice', fontsize=25)
        plt.legend(loc="top right")
        