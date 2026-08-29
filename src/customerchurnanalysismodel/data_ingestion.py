import pandas as pd
import numpy as np 

def data_ingestion():
    df = pd.read_csv(r'https://github.com/deveshdubey18/Customer_Churn_Analysis_Model/raw/refs/heads/main/data/raw/churn_dataset.csv')
    return df