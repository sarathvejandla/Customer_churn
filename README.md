# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
- This project is to identify the customers that are at the risk of churn by building a model which is trained on historical customer data. 

## Files and data description
1. Guide.ipynb : This has instructions provided by Udacity on what's needed to be build/refactored
2. churn_notebook.py : This is a jupyter notebook that has code to build a customer churn model ( import_data(), perform_eda(), encoder_helper(), perform_feature_engineering(), train_models() )
3. churn_library.py : This is a refactored code of churn_nootbook.py
test_churn_script_logging_and_tests.py : This file has test cases to test the functions present in churn_library.py
4. churn_library.log : This is a log file that has results of test & logging information. 
README.md : This has detail on project, files & how to run the code


## Running Files
1. churn_library.py :
        Command to run: 
            1. cd Customer_churn
            2. python churn_library.py
        Action: Running the above command imports data, perform EDA & feature engineering & trains on data and saves the model files. 

2. test_churn_script_logging_and_tests.py :
        Command to run: 
            1. cd Customer_churn
            2. churn_script_logging_and_tests.py
        Action: Running the above command tests all the functions in the churn_library.py and saves results to churn_library.log



