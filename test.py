import os
import logging
import churn_library as cls

def get_dataframe():
    df = cls.import_data("./data/bank_data.csv")
    print(df.head())
    return df

if __name__ == "__main__":
    get_dataframe()