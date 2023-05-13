import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np


#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - Tweet Emoji Recommendation")

#############################################

st.title('Train Model')

#############################################
def split_dataset(df, number, random_state=42):
    """
    This function splits the dataset into the training and test sets.
    Input:
        - X: training features
        - y: training targets
        - number: the ratio of test samples
        - target: article feature name 'rating'
        - feature_encoding: (string) 'Word Count' or 'TF-IDF' encoding
        - random_state: determines random number generation for centroid initialization
    Output:
        - X_train_sentiment: training features (word encoded)
        - X_val_sentiment: test/validation features (word encoded)
        - y_train: training targets
        - y_val: test/validation targets
    """
    # X_train = []
    # X_val = []
    # y_train = []
    # y_val = []
    try:
        
        X, y = df.drop(['Emoji', 'Text'], axis=1), df["Emoji"]
        # Split the train and test sets into X_train, X_val, y_train, y_val using X, y, number/100, and random_state
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=number/100, random_state=random_state)

        # Compute dataset percentages
        train_percentage = (len(X_train) /
                            (len(X_train)+len(X_val)))*100
        test_percentage = (len(X_val) /
                           (len(X_train)+len(X_val)))*100

        # Print dataset split result
        st.markdown('The training dataset contains {0:.2f} observations ({1:.2f}%) and the test dataset contains {2:.2f} observations ({3:.2f}%).'.format(len(X_train),
                                                                                                                                                          train_percentage,
                                                                                                                                                          len(X_val),
                                                                                                                                                          test_percentage))
        # Save train and test split to st.session_state
        st.session_state['X_train'] = X_train
        st.session_state['X_val'] = X_val
        st.session_state['y_train'] = y_train
        st.session_state['y_val'] = y_val
    except:
        print('Exception thrown; testing test size to 0')
    return X_train, X_val, y_train, y_val


def inspect_coefficients(models):
    """
    This function gets the coefficients of the trained models and displays the model name and coefficients

    Input:
        - trained_models: list of trained names (strings)
    Output:
        - out_dict: a dicionary contains the coefficients of the selected models, with the following keys:
            - model_name: model coefficients
    """
    out_dict = {}
    for model_name, model in models.items():
        if (model_name == 'Random Forest'):
            feature_importances = model.feature_importances_
            out_dict[model_name] = feature_importances
            st.write('### Feature Importances for {0}'.format(model_name))
            st.write('Total number of features: {0}'.format(len(feature_importances)))
            st.write('Number of important features: {0}'.format(len(feature_importances[feature_importances > 0])))

        elif (model_name == 'Neural Network (MLP)'):
            coef = model.coefs_ 
            out_dict[model_name] = coef
            total_count = 0
            positive_count = 0
            negative_count = 0

            for layer_weights in coef:
                layer_weights = np.array(layer_weights)
                positive_count += np.count_nonzero(layer_weights > 0)
                negative_count += np.count_nonzero(layer_weights < 0)
                total_count += layer_weights.size

            st.write('### Coefficients inspection for {0}'.format(model_name))
            st.write('Total number of coefficients: {0}'.format(len(coef)))
            for i, c in enumerate(coef):
                st.write(f"Weight matrix {i} shape: {c.shape}")
            st.write('Number of positive coefficients: {0}'.format(positive_count))
            st.write('Number of negative coefficients: {0}'.format(negative_count))
        
        else:
            st.write('Invalid model name provided in inspect_coefficients')
    return out_dict

def train_random_forest(params):
    X_train =  st.session_state['X_train']
    y_train =  st.session_state['y_train']
    # st.write(X_train)
    model = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                   max_depth=params['max_depth'], 
                                   random_state=params['random_state'])
    # y_train = np.array(y_train).reshape(-1, 1)
              
    model.fit(X_train, y_train)
    return model

def train_mlp(params):
    X_train =  st.session_state['X_train']
    y_train =  st.session_state['y_train']
    model = MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'], 
                          activation=params['activation'], 
                          max_iter=params['max_iter'], 
                          random_state=params['random_state'])
    model.fit(X_train, y_train)
    return model
#############################################################################################################

df = None
df = fetch_dataset()

if st.button('Load New Dataset'):
    data = st.file_uploader(
            'Upload a Dataset', type=['csv', 'txt'])
    if (data):
        df = pd.read_csv(data)
    if df is not None:
        st.write("Loaded dataset has {} rows and {} columns.".format(
            df.shape[0], df.shape[1]))

if df is not None:
    # Display dataframe as table
    st.dataframe(df)

    # Split dataset
    st.markdown('### Split dataset into Train/Validation/Test sets')

    # Select word count encoder
    word_count_encoder_options = ['Word Count', 'TF-IDF']
    if ('word_encoder' in st.session_state):
        if (st.session_state['word_encoder'] is not None):
            word_count_encoder_options = st.session_state['word_encoder']
            st.write('Restoring selected encoded features {}'.format(
                word_count_encoder_options))

    # Select input features
    feature_input_select = st.selectbox(
        label='Select features for classification input',
        options=word_count_encoder_options,
        key='feature_select'
    )

    st.session_state['feature'] = feature_input_select

    # Select test size
    st.markdown(
        '#### Enter the percentage of validation/test data to use for training the model')
    number = st.number_input(
        label='Enter size of test set (X%)', min_value=1, max_value=100, value=30, step=1)
    st.write('You selected {}% of the dataset for training'.format(number))
    X_train, X_val, y_train, y_val = [], [], [], []
    # Compute the percentage of test and training data
    if (st.button('Split dataset')):
        X_train, X_val, y_train, y_val = split_dataset(df, number)

    # Train models
    st.markdown('### Train models')
    model_options = ['Random Forest', 'Neural Network (MLP)']

    # Collect ML Models of interests
    model_select = st.multiselect(
        label='Select which model you wish to use for classification predictions',
        options=model_options,
    )
    st.write('You selected the follow models: {}'.format(model_select))


    # Random Forest parameters selection
    if (model_options[0] in model_select):
        st.markdown('#### ' + model_options[0])

        param_col1, param_col2 = st.columns(2)
        with (param_col1):
            param1_options = [100, 200, 300, 400] # Number of trees in the forest
            param1_select = st.selectbox(
                label='Select number of trees in the forest',
                options=param1_options,
                key='param1_select'
            )
            st.write('You select the following number of trees in the forest: {}'.format(param1_select))

            param2_options = [None, 10, 20, 30]  # Maximum depth of the trees
            param2_select = st.selectbox(
                label='Select maximum depth of the trees',
                options=param2_options,
                key='param2_select'
            )
            st.write('You select the following maximum depth of the tress: {}'.format(param2_select))

        with (param_col2):
            param3_options =  [42]  # Random state for reproducibility
            param3_select = st.selectbox(
                label='Select number of random state',
                options=param3_options,
                key='param3_select'
            )
            st.write('You select the following random states: {}'.format(param3_select))

        rf_params = {
            'n_estimators': param1_select,
            'max_depth': param2_select,
            'random_state': param3_select
        }
        
        # MLP parameters selection
        if (model_options[1] in model_select):
            st.markdown('#### ' + model_options[1])

            param_col1, param_col2 = st.columns(2)
            with (param_col1):
                param4_options = [(5,), (100,), (200,), (300,), (400,)]  # Number of neurons in hidden layers
                param4_select = st.selectbox(
                label='Select number of neurons in hidden layers',
                options=param4_options,
                key='param4_select'
                )
                st.write('You select the following number of neurons in hidden layers: {}'.format(param4_select))

                param5_options = ['relu', 'tanh', 'logistic']  # Activation function
                param5_select = st.selectbox(
                    label='Select which activation function to use',
                    options=param5_options,
                    key='param5_options'
                )
                st.write('You select the following activation function: {}'.format(param5_options))

            with (param_col2):
                param6_options = [200, 300, 400]  # Maximum number of iterations
                param6_select = st.selectbox(
                    label='Select number of iterations to perform',
                    options=param6_options,
                    key='param6_options'
                )
                st.write('You select the following max number of iterations: {}'.format(param6_options))

                param7_options = [42]  # Random state for reproducibility
                param7_select = st.selectbox(
                    label='Select number of random states',
                    options=param7_options,
                    key='param7_options'
                )
                st.write('You select the following random state: {}'.format(param7_options))

            nn_params = {
                'hidden_layer_sizes': param4_select,
                'activation': param5_select,
                'max_iter': param6_select,
                'random_state': param7_select
            }
        
        if st.button('Train Random Forest Model'):
            st.session_state[model_options[0]] = train_random_forest(rf_params)
            # st.write('Random Forest Model trained')

        if st.button('Train Neural Network Classifier (MLP) Model'):
            st.session_state[model_options[1]] = train_mlp(nn_params)
            # st.write('Neural Network Classifier (MLP) Model trained')

        if model_options[0] not in st.session_state:
            st.write('Random Forest Model is untrained')
        else:
            st.write('Random Forest Model trained')

        if model_options[1] not in st.session_state:
            st.write('Neural Network Classifier (MLP) Model is untrained')
        else:
            st.write('Neural Network Classifier (MLP) Model trained')

    # Inspect coefficients
    st.markdown('### Inspect model coefficients')

    inspect_models = st.multiselect(
        label='Select models to inspect coefficients',
        options=model_select,
        key='inspect_multiselect'
    )

    models = {}
    for model_name in inspect_models:
        if (model_name in st.session_state):
            models[model_name] = st.session_state[model_name]

    coefficients = inspect_coefficients(models)

    st.write('Continue to Test Model')
