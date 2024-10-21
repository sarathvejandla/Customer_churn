# library doc string
"""
Description: This module takes in customer data and predicts customers that are more likely to churn.
Author: Sarath
Date: 10/16/2024
"""

# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() 

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

os.environ['QT_QPA_PLATFORM']='offscreen'

#logging.basicConfig(
#    filename='./logs/churn_library.log',
#    level=logging.INFO,
#    filemode='w',
#    format='%(name)s - %(levelname)s - %(message)s') 

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    #try:
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    #    logging.info("SUCCESS: Loaded data from provided file path")
    return df
    #except FileNotFoundError:
    #    logging.error("ERROR: Please provide a valid path")
    #    return "Please provide a valid path"

def plot_features(df, feature_name, chart_type):
    '''
    Plots data frame columns
    input:
            df: pandas dataframe
            feature_name: column name to be plotted on chart
            chart_type: type of chart to use for plotting

    output:
            None
    '''
    plt.figure(figsize=(20,10)) 
    if chart_type == 'hist':
        df[feature_name].hist()
    elif chart_type == 'bar':
        df[feature_name].value_counts('normalize').plot(kind='bar')
    elif chart_type == 'histplot':
        sns.histplot(df[feature_name], stat='density', kde=True)

    plt.savefig('./images/eda/'+feature_name+'_distribution.png')
    plt.close()

def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
#    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    plot_features(df, 'Churn', 'hist')
    plot_features(df, 'Customer_Age', 'hist')
    plot_features(df, 'Marital_Status', 'bar')
    plot_features(df, 'Total_Trans_Ct', 'histplot')

    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('./images/eda/correlation_chart.png')
    plt.close()

 
def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    for category in category_lst:
        category_groupby_lst = []
        category_groups = df.groupby(category).mean()[response]
        for val in df[category]:
                category_groupby_lst.append(category_groups.loc[val])
        df[category+'_Churn'] = category_groupby_lst
    #print(df.columns)
    return df
    

def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
    ]

    quant_columns = [
    'Customer_Age',
    'Dependent_count', 
    'Months_on_book',
    'Total_Relationship_Count', 
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 
    'Credit_Limit', 
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 
    'Total_Amt_Chng_Q4_Q1', 
    'Total_Trans_Amt',
    'Total_Trans_Ct', 
    'Total_Ct_Chng_Q4_Q1', 
    'Avg_Utilization_Ratio'
    ]
    
    encoded_cust_data = encoder_helper(df, cat_columns, response)

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    
    features = pd.DataFrame()
    features[keep_cols] = encoded_cust_data[keep_cols]
    label = encoded_cust_data[response]

    features_train, features_test, label_train, label_test = train_test_split(features, label, test_size= 0.3, random_state=42)

    return features_train, features_test, label_train, label_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    y_preds = [[y_train_preds_rf,y_test_preds_rf], [y_train_preds_lr, y_test_preds_lr]]
    models = ['Random Forest', 'Logistic Regression']

    for i,j in zip(y_preds,models):
            plt.rc('figure', figsize=(5, 5))
            plt.text(0.01, 1.25, str(j+'Train'), {'fontsize': 10}, fontproperties = 'monospace')
            plt.text(0.01, 0.05, str(classification_report(y_train, i[0])), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
            plt.text(0.01, 0.6, str(j+'Test'), {'fontsize': 10}, fontproperties = 'monospace')
            plt.text(0.01, 0.7, str(classification_report(y_test, i[1])), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
            plt.axis('off')
            plt.savefig('./images/results/'+j+'_classification_report.png')
            plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth+'feature_importance_plot.png')
    plt.close()

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)
    
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)

    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("./images/results/roc_curve.png")
    plt.close()

    feature_importance_plot(cv_rfc.best_estimator_, X_train, "./images/results/")

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    
    customer_data = import_data("./data/bank_data.csv")

    perform_eda(customer_data)
    
    features_train, features_test, label_train, label_test = perform_feature_engineering(customer_data, 'Churn')
    
    train_models(features_train, features_test, label_train.values.ravel(), label_test.values.ravel())



