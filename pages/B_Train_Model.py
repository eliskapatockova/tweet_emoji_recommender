import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - <project title>")

#############################################

st.title('Train Model')

#############################################
df = None
df = fetch_dataset()

def split_dataset(df, number, feature_encoding, random_state=42):
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
    X_train = []
    X_val = []
    y_train = []
    y_val = []

    X_train_emoji, X_val_emoji = [], []
    try:
        X, y = df.drop(['Emoji', 'Emoji_Labels', 'Tweet'], axis=1), df["Emoji_Labels"]

        # Split the train and test sets into X_train, X_val, y_train, y_val using X, y, number/100, and random_state
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=number/100, random_state=random_state)

        # Use the column word_count as a feature and the column sentiment as the target
        if ('TF-IDF' in feature_encoding):
            X_train_emoji = X_train.loc[:, X_train.columns.str.startswith(
                'tf_idf_word_count_')]
            X_val_emoji = X_val.loc[:, X_val.columns.str.startswith(
                'tf_idf_word_count_')]
        elif ('Word Count' in feature_encoding):
            X_train_emoji = X_train.loc[:, X_train.columns.str.startswith(
                'word_count_')]
            X_val_emoji = X_val.loc[:,
                                        X_val.columns.str.startswith('word_count_')]
        else:
            st.write('Invalid feature encoding provided in split_dataset')

        # Compute dataset percentages
        train_percentage = (len(X_train) /
                            (len(X_train)+len(X_val)))*100
        test_percentage = (len(X_val) /
                           (len(X_train)+len(X_val)))*100

        # Print dataset split result
        st.markdown('The training dataset contains {0:.2f} observations ({1:.2f}%) and the test dataset contains {2:.2f} observations ({3:.2f}%).'.format(len(X_train),
                                                                                                                                                          train_percentage,
                                                                                                                                                          len(
                                                                                                                                                              X_val),
                                                                                                                                                          test_percentage))

        # Save train and test split to st.session_state
        st.session_state['X_train'] = X_train_emoji
        st.session_state['X_val'] = X_val_emoji
        st.session_state['y_train'] = y_train
        st.session_state['y_val'] = y_val
    except:
        print('Exception thrown; testing test size to 0')
    return X_train_emoji, X_val_emoji, y_train, y_val


def inspect_coefficients(models):
    pass

def random_forest(X_train, X_val, y_train):
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_val)
    return classifier, y_pred

def mlp(X_train, X_val, y_train):
    nn_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    nn_classifier.fit(X_train, y_train)

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
        label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

    X_train, X_val, y_train, y_val = [], [], [], []
    # Compute the percentage of test and training data
    if (st.button('Split dataset')):
        X_train, X_val, y_train, y_val = split_dataset(df, number, feature_input_select)

    # Train models
    st.markdown('### Train models')
    model_options = ['Random Forest', 'MLP Classifier']

    # Collect ML Models of interests
    model_select = st.multiselect(
        label='Select which model you wish to use for classification predictions',
        options=model_options,
    )
    st.write('You selected the follow models: {}'.format(model_select))

    if (model_options[0] in model_select):
        st.markdown('#### ' + model_options[0])

        param_col1, param_col2 = st.columns(2)
        with (param_col1):
            param1_options = []
            param1_select = st.selectbox(
                label='Select param1',
                options=param1_options,
                key='param1_select'
            )
            st.write('You select the following <param1>: {}'.format(param1_select))

            param2_options = []
            param2_select = st.selectbox(
                label='Select param2',
                options=param2_options,
                key='param2_select'
            )
            st.write('You select the following <param2>: {}'.format(param2_select))

        with (param_col2):
            param3_options = []
            param3_select = st.selectbox(
                label='Select param3',
                options=param3_options,
                key='param3_select'
            )
            st.write('You select the following <param3>: {}'.format(param3_select))

            param4_options = []
            param4_select = st.selectbox(
                label='Select param4',
                options=param4_options,
                key='param4_select'
            )
            st.write('You select the following <param4>: {}'.format(param4_select))

        model_params = {
            'param1': param1_select,
            'param2': param2_select,
            'param3': param3_select,
            'param4': param4_select
        }

        if st.button('Train Random Forest Model'):
            random_forest(
                X_train, y_train, model_options[0], rf_params)

        if model_options[0] not in st.session_state:
            st.write('Random Forest Model is untrained')
        else:
            st.write('Random Forest Model trained')

        if st.button('Train Neural Network Classifier (MLP) Model'):
            random_forest(
                X_train, y_train, model_options[1], nn_params)

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
