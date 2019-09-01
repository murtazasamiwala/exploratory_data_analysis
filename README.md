# exploratory_data_analysis
 python class for EDA
 Class to conduct exploratory analysis of dataframe (df) for a given dependent 
    variable (dep) and indepent quan variables (indq). If independent quan variables 
    (indq) not given, selects all int/float columns apart from dependent variable as 
    independent quan variables
    
    For class object of given dataframe (df), dependent variable (dep), and
    independent quantitative variables (indq, optional), following methods available:
    1. boxplots: Provides side-by-side boxplots.
    2. snheatmap: Provides correlation heatmap.
    3. scatter: Provides scatter plot with dependent variable.
    4. snpairplot: Provides seaborn pairplots.
    5. normality_test: Provides normality test plots for specified columns.
    6. outlier_info: Provides number of outliers.
    
    See method docstrings for more information.
