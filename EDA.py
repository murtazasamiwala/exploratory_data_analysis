#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mth
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from scipy.signal import argrelextrema
from scipy.stats import shapiro, normaltest
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm

# In[1]:


class Explore:
    '''Class to conduct exploratory analysis of dataframe (df) for a given dependent 
    variable (dep) and indepent quan variables (indq). If independent quan variables 
    (indq) not given, selects all int/float columns apart from dependent variable as 
    independent quan variables'''
    info='''For class object of given dataframe (df), dependent variable (dep), and
    independent quantitative variables (indq, optional), following methods available:
    1. boxplots: Provides side-by-side boxplots.
    2. snheatmap: Provides correlation heatmap.
    3. scatter: Provides scatter plot with dependent variable.
    4. snpairplot: Provides seaborn pairplots.
    5. normality_test: Provides normality test plots for specified columns.
    6. outlier_info: Provides number of outliers.
    
    See method docstrings for more information.'''
    
    
    def __init__(self,df,dep,indq=None):
        '''Get df, dependent column name (dep, as string), and independent quantitative 
        column (indq, as list,default None). If independent quanitative columns not given, 
        int and float columns selected automatially'''
        self.df=df
        self.dep=dep
        self.q=[]
        for i in self.df.columns:
            if self.df[i].dtype in [int,'int64',float] and i!=dep:
                self.q.append(i)
        self.indq=indq if indq is not None else self.q
        return print(self.info)
    
    def boxplots(self,cols=None):
        '''Plots side-by-side boxplots for specified columns (cols, as list or string). If 
        columns not specified, plots for all independent quan variables'''
        self.cols=cols if cols is not None and type(cols) is list else [cols] if cols is not None and type(cols) is str else self.indq 
        fig, axes=plt.subplots(1,len(self.indq),figsize=(22,8),sharey=False)
        for i in range(len(self.cols)):
            axes[i].boxplot(self.df[self.cols[i]],meanline=True, showmeans=True, showcaps=True,showbox=True,showfliers=True)
            axes[i].set(title=self.cols[i])
        return
    
    def snheatmap(self,cols=None):
        '''Plots seaborn heatmap for specified columns (cols, as list or string). If columns 
        not specified, plots for all independent quan columns'''
        self.cols=cols if cols is not None and type(cols) is list else [cols] if cols is not None and type(cols) is str else self.indq
        plt.figure(figsize=(10,7))
        sns.heatmap(self.df[self.cols].corr(),annot=True)
        plt.show()
        return
    
    def scatter(self,cols=None):
        '''Plots scatter plot between specified columns (cols, as list or string) and dependent 
        variable. If columns not specified, plots for all independent quan columns'''
        self.cols=cols if cols is not None and type(cols) is list else [cols] if cols is not None and type(cols) is str else self.indq
        fig, axes=plt.subplots(len(self.indq),1,figsize=(8,40))
        for i in range(len(self.cols)):
            sns.regplot(self.df[self.cols[i]],self.df[self.dep],scatter=True,scatter_kws={"color": "green"}, line_kws={"color": "red"},ax=axes[i])
            axes[i].set(xlabel=self.cols[i],ylabel=self.dep,title='Scatter of {} and {}'.format(self.cols[i],self.dep))
        return
    
    def snpairplot(self,cols=None):
        '''Creates pairplots between each pair of specified columns (cols, as list or string). If 
        columns not specified, plots fo each pair of independent quan variables'''
        self.cols=cols if cols is not None and type(cols) is list else [cols] if cols is not None and type(cols) is str else self.indq
        sns.pairplot(self.df[self.cols],diag_kind='kde')
        plt.show()
        return
    
    def normality_plots(self,col):
        'Plots tests of normality for given column in the df'
        fig = plt.figure(figsize=(15, 5))
        shapiro_p = round(shapiro(self.df[col])[1], 2)
        normaltest_p = round(normaltest(self.df[col])[1], 2)
        plt.subplot(1, 3, 1)
        plt.title('Histogram for '+col, color='navy', fontsize=12)
        plt.hist(self.df[col])
        plt.subplot(1, 3, 2)
        plt.title('Q-Q Plot for '+col, color='brown', fontsize=12)
        qqplot(self.df[col], line='s', ax=plt.subplot(1, 3, 2))
        plt.subplot(1, 3, 3)
        plt.title('Normality Test Results for '+col, color='olive', fontsize=12)
        plt.plot([shapiro_p, normaltest_p], linestyle=' ', marker='x')
        plt.text(x=0.2, y=0.5, s='Shapiro\np value\n'+str(shapiro_p))
        plt.text(x=0.6, y=0.5, s='Normaltest\np value\n'+str(normaltest_p))
        plt.ylim((0, 1))
        plt.hlines(y=0.05, color='r', xmin=0, xmax=1)
        plt.suptitle('Normality Test for '+col, fontsize=16, color='b')
        plt.show()
        return
    
    def normality_test(self,cols=None):
        '''Plots normality test plots for specified columns (cols, as list or string). If columns 
        not specified, plots for all independent quan variables'''
        self.cols=cols if cols is not None and type(cols) is list else [cols] if cols is not None and type(cols) is str else self.indq
        for i in range(len(self.cols)):
            self.normality_plots(self.cols[i])
        return
    
    def outlier_info(self,cols=None):
        '''Provides dataframe with outliers numbers for specified columns (cols, as list or string). If
        columns not specified, provides infor for all independent quan variables. Provides outliers
        according to Tukey's 1.5 and 3 fences'''
        self.cols=cols if cols is not None and type(cols) is list else [cols] if cols is not None and type(cols) is str else self.indq
        Q1=self.df[self.cols].quantile(0.25)
        Q3=self.df[self.cols].quantile(0.75)
        IQR=Q3-Q1
        outlier_dict={}
        outlier_dict['''Tukey's 1.5''']=((self.df[self.cols] < (Q1 - 1.5* IQR)) |(self.df[self.cols] > (Q3 + 1.5 * IQR))).sum()
        outlier_dict['''Tukey's 3''']=((self.df[self.cols] < (Q1 - 3* IQR)) |(self.df[self.cols] > (Q3 + 3* IQR))).sum()
        return pd.DataFrame(outlier_dict)
    
def null_check(df,drop_thresh=90):
    null_series=(df.isnull().sum()/len(df))*100
    if df.isnull().sum().sum()==0:
        print("No nulls")
    else:
        treat_series=null_series[(null_series>0)&(null_series<drop_thresh)]
        return treat_series,list(null_series[null_series>=90])    



class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        self.drop_list=[]
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)
            self.drop_list.append(colX)

        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)
    
    def Testall(self,colY):
        for i in self.df.columns:
            self.TestIndependence(i,colY)
        return self.drop_list


