# library doc string
'''
Carry out churn procedure
Produces images in results, models and eda folders
richard edwards 16/09/2021
'''
import os
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM']='offscreen'

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    d_f = pd.read_csv(pth)
    d_f['Churn'] = d_f['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    return d_f


def perform_eda(d_f):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(10,5))
    d_f['Churn'].hist()
    plt.savefig(fname="images/eda/Churn_hist.jpg")
    d_f['Customer_Age'].hist()
    plt.savefig(fname="images/eda/Customer_age_hist.jpg")
    d_f.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname="images/eda/Marital_Status_counts.jpg")
    sns.distplot(d_f['Total_Trans_Ct'])
    plt.savefig(fname="images/eda/Total_Trans_Ct.jpg")
    sns.heatmap(d_f.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig(fname="images/eda/Corr_matrix.jpg")
    plt.close()
    
def encoder_helper(d_f, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with
    cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for name in category_lst:
        group = d_f.groupby(name).mean()[response]
        lst = []
        for val in d_f[name]:
            lst.append(group.loc[val])
        d_f[name + '_Churn'] = lst
    return d_f
    
def perform_feature_engineering(d_f, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be
              used for naming variables or index y column]
    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    x = pd.DataFrame()
    y = d_f[response]
    d_f.drop(columns=['Unnamed: 0','CLIENTNUM','Churn','Gender','Education_Level',
                     'Marital_Status', 'Income_Category', 'Card_Category'],inplace=True)
    x = d_f.copy(deep=True)
    return train_test_split(x, y, test_size= 0.3, random_state=42)    
    
def classification_report_image(data):
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
    y_train = data[0]
    y_test = data[1]
    y_train_preds_lr = data[2]
    y_train_preds_rf = data[3]
    y_test_preds_lr = data[4]
    y_test_preds_rf = data[5]

    classification_reports_data = {
        "Random_Forest": (
            "Random Forest Train",
            y_test,
            y_test_preds_rf,
            "Random Forest Test",
            y_train,
            y_train_preds_rf),
        "Logistic_Regression": (
            "Logistic Regression Train",
            y_train,
            y_train_preds_lr,
            "Logistic Regression Test",
            y_test,
            y_test_preds_lr)}
    
    for title, classification_data in classification_reports_data.items():
        plt.rc("figure", figsize=(8, 8))
        plt.text(0.01, 1.25, str(classification_data[0]), {
                 "fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    classification_data[1], classification_data[2])), {
                "fontsize": 10}, fontproperties="monospace")
        plt.text(0.01, 0.6, str(classification_data[3]), {
                 "fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    classification_data[4], classification_data[5])), {
                "fontsize": 10}, fontproperties="monospace")
        plt.axis("off")
        plt.savefig("images/results/%s.jpg" % title)
        plt.close()
    
def feature_importance_plot(model, x_data, output_pth):
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
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    plt.figure(figsize=(20,5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig("images/%s/Feature_Importance.jpg" % output_pth)
    plt.close()
    
def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
       # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()
    param_grid = { 
        'n_estimators': [25],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [5,10],
        'criterion' :['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)
    
    classification_report_image([y_train,
                            y_test,
                            y_train_preds_lr,
                            y_train_preds_rf,
                            y_test_preds_lr,
                            y_test_preds_rf])
    
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    plt.figure(figsize=(15, 8))
    a_x=plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test, ax=a_x, alpha=0.8)
    lrc_plot.plot(ax=a_x, alpha=0.8)
    plt.savefig(fname="images/results/roc_curve1.jpg")
    plt.close()
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    feature_importance_plot(cv_rfc, x_test, "results")
    
if __name__ == "__main__":
    PATH = "./data/bank_data.csv"
    dataframe = import_data(PATH)
    perform_eda(dataframe)
    dataframe = dataframe.drop(columns =['Attrition_Flag'])
    categories = list(dataframe.select_dtypes(include='object').columns)
    dataframe = encoder_helper(dataframe, categories, response='Churn')
    x_train, x_test, y_train, y_test = perform_feature_engineering(dataframe, response='Churn')
    train_models(x_train, x_test, y_train, y_test)
    