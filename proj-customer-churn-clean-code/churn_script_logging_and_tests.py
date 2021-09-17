'''
Testing file for chirn_library.py
richard edwards 16/09/2021
'''
import os
import logging

import pytest
import joblib

import churn_library

os.environ['QT_QPA_PLATFORM']='offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

@pytest.fixture(name='dataframe_raw')
def dataframe_raw_():
    """
    raw dataframe fixture - returns the raw dataframe from initial dataset file
    """
    try:
        internal_dataframe_raw = churn_library.import_data(
            "data/bank_data.csv")
        logging.info("Raw dataframe fixture creation: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Raw dataframe fixture creation: The file wasn't found")
        raise err
    return internal_dataframe_raw


@pytest.fixture(name='dataframe_encoded')
def dataframe_encoded_(dataframe_raw):
    """
    encoded dataframe fixture - returns the encoded dataframe on some specific column
    """
    dataframe_raw = dataframe_raw.drop(columns =['Attrition_Flag'])
    categories = list(dataframe_raw.select_dtypes(include='object').columns)
    
    try:
        dataframe_encoded = churn_library.encoder_helper(
            dataframe_raw, categories,response="Churn")
        logging.info("Encoded dataframe fixture creation: SUCCESS")
    except KeyError as err:
        logging.error(
            "Encoded dataframe fixture creation: Not existent column to encode")
        raise err
    return dataframe_encoded


@pytest.fixture(name='feature_sequences')
def feature_sequences_(dataframe_encoded):
    """
    feature sequences fixtures - returns 4 series containing features sequences
    """
    try:
        x_train, x_test, y_train, y_test = churn_library.perform_feature_engineering(
            dataframe_encoded,response='Churn')

        logging.info("Feature sequence fixture creation: SUCCESS")
    except BaseException:
        logging.error(
            "Feature sequences fixture creation: Sequences length mismatch")
        raise
    return x_train, x_test, y_train, y_test


def test_import(dataframe_raw):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''

    try:
        assert dataframe_raw.shape[0] > 0
        assert dataframe_raw.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
  


def test_eda(dataframe_raw):
    '''
    test perform eda function
    '''
    churn_library.perform_eda(dataframe_raw)
    figures = ["Churn_hist", "Customer_age_hist", "Marital_Status_counts", "Total_Trans_Ct","Corr_matrix"]
    for image_name in figures:
        try:
            with open("./images/eda/%s.jpg" % image_name, 'r'):
                logging.info("Testing perform_eda: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing perform_eda: generated images missing")
            raise err


def test_encoder_helper(dataframe_encoded):
    '''
    test encoder helper
    '''
    columns = ["Gender","Education_Level","Marital_Status","Income_Category","Card_Category"]
    try:
        assert dataframe_encoded.shape[0] > 0
        assert dataframe_encoded.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't appear to have rows and columns")
        raise err
    try:
        for column in columns:
            assert column in dataframe_encoded
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't have the right encoded columns")
        raise err
    logging.info("Testing encoder_helper: SUCCESS")
    return dataframe_encoded

    

def test_perform_feature_engineering(feature_sequences):
    '''
    test perform_feature_engineering
    '''
    x_train = feature_sequences[0]
    x_test = feature_sequences[1]
    y_train = feature_sequences[2]
    y_test = feature_sequences[3]
   
    
    try:
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
    except AssertionError as err:
        logging.error("Testing test_perform_feature_engineering: Failure")
        raise err
    return feature_sequences
        

def test_train_models(feature_sequences):
    """
    test train_models - check result of training process
    """
    churn_library.train_models(
        feature_sequences[0],
        feature_sequences[1],
        feature_sequences[2],
        feature_sequences[3])
    try:
        joblib.load('models/rfc_model.pkl')
        joblib.load('models/logistic_model.pkl')
        logging.info("Testing testing_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: No files found")
        raise err
        
    names = ["Logistic_Regression","Random_Forest","Feature_Importance","roc_curve1"]
    
    for image_name in names:
        try:
            with open("images/results/%s.jpg" % image_name, 'r'):
                logging.info("Testing testing_models (report generation): SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing testing_models (report generation): generated images missing")
            raise err