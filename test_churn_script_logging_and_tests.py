import os
import logging
from churn_library import import_data, perform_feature_engineering, perform_eda, encoder_helper, train_models
import pytest

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture()
def get_dataframe():
    df = import_data("./data/bank_data.csv")
    return df

@pytest.fixture()
def get_train_test(get_dataframe):
    features_train, features_test, label_train, label_test = perform_feature_engineering(
        get_dataframe, 'Churn')
    return features_train, features_test, label_train, label_test


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(get_dataframe):
    '''
    test perform eda function

    input: 
        get_dataframe fixture that has customer data

    output:
        None
    '''

    perform_eda(get_dataframe)
    try:
        assert os.path.isfile('./images/eda/correlation_chart.png')
        logging.info("Testing EDA function : EDA files saved : SUCCESS")
    except AssertionError:
        logging.error(
            "Testing EDA function : EDA files not saved successfully : ERROR")


def test_encoder_helper(get_dataframe):
    '''
    test encoder helper

    input:
        get_dataframe fixture that has customer data

    output: None
    '''
    #df = pytest.fixture("get_dataframe")

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    encoded_columns = [
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn'
    ]

    try:
        encoded_df = encoder_helper(get_dataframe, cat_columns, 'Churn')
#		assert sum([encoded_columns in encoded_df.columns for column_name in encoded_columns]) == 5
        assert len(
            [col for col in encoded_columns if col in encoded_df.columns]) == 5
        logging.info(
            "Testing encoder helper function: encoded columns present in df: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder helper function: encoded columns not present in df: ERROR")
        return err

    for col in encoded_columns:
        try:
            assert encoded_df[col].shape[0] > 0
            logging.info(
                "Testing encoder_helper function: %s field have data : SUCCESS", col)
        except AssertionError as err:
            logging.error(
                "Testing encoder_helper function: %s field doesn't appear to have data : ERROR",
                col)
            raise err


def test_perform_feature_engineering(get_dataframe):
    '''
    test perform_feature_engineering

    input: get_dataframe fixture that has customer data

    output: None
    '''
    df = get_dataframe

    try:
        features_train, features_test, label_train, label_test = perform_feature_engineering(
            get_dataframe, 'Churn')
        logging.info("Feature engineering: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Feature engineering: perform_feature_engineering function failed to run : ERROR")
        raise err

    try:
        assert features_train.shape[0] > 0
        assert features_train.shape[1] > 0
        logging.info("Feature engineering: features_train have data : SUCCESS")
    except AssertionError as err:
        logging.error(
            "Feature engineering: features_train doesn't appear to have rows and columns : ERROR")
        raise err

    try:
        assert features_test.shape[0] > 0 and features_test.shape[1] > 0
        logging.info("Feature engineering: feature_test have data : SUCCESS")
    except AssertionError as err:
        logging.error(
            "Feature engineering: feature_test doesn't appear to have rows and columns : ERROR")
        raise err

    try:
        assert label_train.shape[0] > 0
        logging.info("Feature engineering: label_train have data : SUCCESS")
    except AssertionError as err:
        logging.error(
            "Feature engineering: label_train file doesn't appear to have rows and columns : ERROR")
        raise err

    try:
        assert label_test.shape[0] > 0
        logging.info("Feature engineering: label_test have data : SUCCESS")
    except AssertionError as err:
        logging.error(
            "Feature engineering: label_test file doesn't appear to have rows and columns : ERROR")
        raise err


def test_train_models(get_train_test):
    '''
    test train_models

    input:
        get_train_test fixture

    output: None
    '''
    features_train, features_test, label_train, label_test = get_train_test
    train_models(features_train, features_test, label_train, label_test)

    try:
        assert os.path.isfile(
            './images/results/Logistic Regression_classification_report.png')
        logging.info(
            "Train model : Logistic Regression_classification_report.png file saved : SUCCESS")
    except AssertionError:
        logging.error(
            "Train model : Logistic Regression_classification_report.png file not saved : ERROR")

    try:
        assert os.path.isfile(
            './images/results/Random Forest_classification_report.png')
        logging.info(
            "Train model : Random Forest_classification_report.png file saved : SUCCESS")
    except AssertionError:
        logging.error(
            "Train model : Random Forest_classification_report.png file not saved : ERROR")

    try:
        assert os.path.isfile('./images/results/roc_curve.png')
        logging.info("Train model : roc_curve.png file saved : SUCCESS")
    except AssertionError:
        logging.error("Train model : roc_curve.png file not saved : ERROR")

    try:
        assert os.path.isfile('./images/results/feature_importance_plot.png')
        logging.info(
            "Train model : feature_importance_plot.png file saved : SUCCESS")
    except AssertionError:
        logging.error(
            "Train model : feature_importance_plot.png file not saved : ERROR")

    try:
        assert os.path.isfile('./models/logistic_model.pkl')
        logging.info("Train model : logistic_model.pkl file saved : SUCCESS")
    except AssertionError:
        logging.error(
            "Train model : logistic_model.pkl file not saved : ERROR")

    try:
        assert os.path.isfile('./models/rfc_model.pkl')
        logging.info("Train model : rfc_model.pkl file saved : SUCCESS")
    except AssertionError:
        logging.error("Train model : rfc_model.pkl file not saved : ERROR")


if __name__ == "__main__":
    pytest.main()
