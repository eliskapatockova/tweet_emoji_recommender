import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

# All pages


def fetch_dataset():
    """
    This function renders the file uploader that fetches the dataset either from local machine

    Input:
        - page: the string represents which page the uploader will be rendered on
    Output: None
    """
    # Check stored data
    df = None
    data = None
    if 'data' in st.session_state:
        df = st.session_state['data']
    else:
        data = st.file_uploader(
            'Upload a Dataset', type=['csv', 'txt'])

        if (data):
            df = pd.read_csv(data)
    if df is not None:
        st.session_state['data'] = df
    return df


def compute_precision(y_true, y_pred):
    """
    Measures the precision between predicted and actual values

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - precision score
    """
    precision = -1
    precision = precision_score(y_true, y_pred)
    return precision


def compute_recall(y_true, y_pred):
    """
    Measures the recall between predicted and actual values

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - recall score
    """
    recall = -1
    recall = recall_score(y_true, y_pred)
    return recall


def compute_accuracy(y_true, y_pred):
    """
    Measures the accuracy between predicted and actual values

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - accuracy score
    """
    accuracy = -1
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def compute_f1_score(y_true, y_pred):
    """
    Measures the f1 score between predicted and actual values

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - f1 score
    """
    f1 = -1
    f1 = f1_score(y_true, y_pred)
    return f1