import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset, compute_metrics, compute_accuracy, apply_threshold
from pages.B_Train_Model import split_dataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import recall_score, precision_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - Tweet Emoji Recommendation")

#############################################

st.title('Test Model')
#############################################

# Helper Functions
def compute_eval_metrics(X, y_true, model, model_name):
    """
    This function computes one or more metrics (precision, recall, accuracy) using the model

    Input:
        - X: pandas dataframe with training features
        - y_true: pandas dataframe with true targets
        - model: the model to evaluate
        - metrics: the metrics to evaluate performance (string); 'precision', 'recall', 'accuracy'
    Output:
        - metric_dict: a dictionary contains the computed metrics of the selected model, with the following structure:
            - {metric1: value1, metric2: value2, ...}
    """
    metric_dict = {'precision': -1,
                   'recall': -1,
                   'accuracy': -1,
                   'f1_score': -1
                   }

    # Predict the product sentiment using the input model and data X
    y_pred = model.predict(X)

    # Compute the evaluation metrics in 'metrics = ['precision', 'recall', 'accuracy']' using the predicted sentiment
    if model_name == "Neural Network (MLP)":
        metric_dict['precision'], metric_dict['recall'], metric_dict['f1_score'] = compute_metrics(
            y_true, y_pred)
        metric_dict['accuracy'] = compute_accuracy(y_true, y_pred)
    elif model_name == "Random Forest":
        metric_dict['precision'], metric_dict['recall'], metric_dict['f1_score'] = compute_metrics(
            y_true, y_pred)
        metric_dict['accuracy'] = compute_accuracy(y_true, y_pred)

    return metric_dict

def restore_data_splits(df):
    """
    This function restores the training and validation/test datasets from the training page using st.session_state
                Note: if the datasets do not exist, re-split using the input df

    Input: 
        - df: the pandas dataframe
    Output: 
        - X_train: the training features
        - X_val: the validation/test features
        - y_train: the training targets
        - y_val: the validation/test targets
    """
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    # Restore train/test dataset
    if ('X_train' in st.session_state):
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        st.write('Restored train data ...')
    if ('X_val' in st.session_state):
        X_val = st.session_state['X_val']
        y_val = st.session_state['y_val']
        st.write('Restored test data ...')
    if (X_train is None):
        # Select variable to explore
        numeric_columns = list(df.select_dtypes(include='number').columns)
        feature_select = st.selectbox(
            label='Select variable to predict',
            options=numeric_columns,
        )
        X = df.loc[:, ~df.columns.isin([feature_select])]
        Y = df.loc[:, df.columns.isin([feature_select])]

        # Split train/test
        st.markdown(
            '### Enter the percentage of test data to use for training the model')
        number = st.number_input(
            label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

        X_train, X_val, y_train, y_val = split_dataset(X, Y, number, feature_select, 'TF-IDF')
        st.write('Restored training and test data ...')
    return X_train, X_val, y_train, y_val

def plot_roc_curve(X_train, X_val, y_train, y_val, trained_models, model_names):
    """
    Plot the ROC curve between predicted and actual values for model names in trained_models on the training and validation datasets

    Input:
        - X_train: training input data
        - X_val: test input data
        - y_true: true targets
        - y_pred: predicted targets
        - trained_model_names: trained model names
        - trained_models: trained models in a dictionary (accessed with model name)
    Output:
        - fig: the plotted figure
        - df: a dataframe containing the train and validation errors, with the following keys:
            - df[model_name.__name__ + " Train Precision"] = train_precision_all
            - df[model_name.__name__ + " Train Recall"] = train_recall_all
            - df[model_name.__name__ + " Validation Precision"] = val_precision_all
            - df[model_name.__name__ + " Validation Recall"] = val_recall_all
    """
    # Set up figures
    fig = make_subplots(rows=len(trained_models), cols=1,
                        shared_xaxes=True, vertical_spacing=0.1)

    # Intialize variables
    df = pd.DataFrame()
    threshold_values = np.linspace(0.5, 1, num=100)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    for i, trained_model in enumerate(trained_models):
        model_name = model_names[i]

        train_precision_all = []
        train_recall_all = []
        val_precision_all = []
        val_recall_all = []
        train_probabilities = []
        val_probabilities = []

        # Make predictions on the train and test set using predict_proba() function
        train_probabilities = trained_model.predict_proba(X_train)
        val_probabilities = trained_model.predict_proba(X_val)

        # Computer precision and recall on training set using threshold_values
        # Computer precision and recall on validation set using threshold_values
        for threshold in threshold_values:
            train_predictions = apply_threshold(
                train_probabilities, threshold)
            val_predictions = apply_threshold(val_probabilities, threshold)

            precision = precision_score(
                y_train_encoded, train_predictions, zero_division=1)
            recall = recall_score(y_train_encoded, train_predictions)
            train_precision_all.append(precision)
            train_recall_all.append(recall)

            precision = precision_score(
                y_val_encoded, val_predictions, zero_division=1)
            recall = recall_score(y_val, val_predictions)
            val_precision_all.append(precision)
            val_recall_all.append(recall)

        # Plot ROC Curve
        fig.add_trace(go.Scatter(x=train_recall_all,
                                 y=train_precision_all, name="Train"), row=i+1, col=1)

        fig.add_trace(go.Scatter(x=val_recall_all,
                                 y=val_precision_all, name="Validation"), row=i+1, col=1)

        fig.update_xaxes(title_text="Recall")
        fig.update_yaxes(title_text='Precision', row=i+1, col=1)
        fig.update_layout(title=model_name+' ROC Curve')
        # Save output values
        df[model_name+" Train Precision"] = train_precision_all
        df[model_name+" Train Recall"] = train_recall_all
        df[model_name+" Validation Precision"] = val_precision_all
        df[model_name+" Validation Recall"] = val_recall_all
    return fig, df


#############################################

df = None
df = fetch_dataset()

if df is not None:
    X_train, X_val, y_train, y_val = restore_data_splits(df)
    st.markdown("### Get Performance Metrics")
    metric_options = ['precision', 'recall', 'accuracy', 'f1_score']
    model_options = ['Random Forest', 'Neural Network (MLP)']
    trained_models = [model for model in model_options if model in st.session_state]
    st.session_state['trained_models'] = trained_models

    # Select a trained classification model for evaluation
    model_select = st.multiselect(
        label='Select trained models for evaluation',
        options=trained_models
    )

    if (model_select):
        st.write(
            'You selected the following models for evaluation: {}'.format(model_select))

        eval_button = st.button('Evaluate your selected classification models')

        if eval_button:
            st.session_state['eval_button_clicked'] = eval_button

        if 'eval_button_clicked' in st.session_state and st.session_state['eval_button_clicked']:
            st.markdown('### Review Model Performance')

            review_options = ['ROC Curve', 'metrics']

            review_plot = st.multiselect(
                label='Select plot option(s)',
                options=review_options
            )

            if 'ROC Curve' in review_plot:
                trained_select = [st.session_state[model]
                                  for model in model_select]
                fig, df = plot_roc_curve(
                    X_train, X_val, y_train, y_val, trained_select, model_select)
                st.plotly_chart(fig)

            if 'metrics' in review_plot:
                models = [st.session_state[model]
                          for model in model_select]

                train_result_dict = {}
                val_result_dict = {}


                for idx, model in enumerate(models):
                    model_name = model_select[idx]
                    st.write(model_name)
                    train_result_dict[model_select[idx]] = compute_eval_metrics(
                        X_train, y_train, model, model_name)
                    val_result_dict[model_select[idx]] = compute_eval_metrics(
                        X_val, y_val, model, model_name)

                st.markdown('### Predictions on the training dataset')
                st.dataframe(train_result_dict)

                st.markdown('### Predictions on the validation dataset')
                st.dataframe(val_result_dict)

    # Select a model to deploy from the trained models
    st.markdown("### Choose your Deployment Model")
    model_select = st.selectbox(
        label='Select the model you want to deploy',
        options=trained_models,
    )

    if (model_select):
        st.write('You selected the model: {}'.format(model_select))
        st.session_state['deploy_model'] = st.session_state[model_select]

    st.write('Continue to Deploy Model')
