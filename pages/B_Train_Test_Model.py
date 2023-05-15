import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, hamming_loss
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - Tweet Emoji Recommendation")

#############################################

st.title('Train Model')

#############################################
df = None

# Split dataset
st.markdown('### Split dataset into Train/Validation/Test sets')
st.markdown('#### Enter the percentage of validation/test data to use for training the model')
number = st.number_input(label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

encodings = ["word_encoded_data", "tfidf_encoded_data"]
models = [("BinaryRelevance", BinaryRelevance), ("ClassifierChain", ClassifierChain), ("LabelPowerset", LabelPowerset)]
st.session_state["trained"] = False
if (st.button('Train Models')):
    st.markdown('### Train Models')
    for encoding in encodings:
        df = st.session_state[encoding]
        X, y = df.loc[:, df.columns.str.startswith('word_count_')], df.loc[:, df.columns.str.startswith('emoji_')]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=number/100, random_state=42)

        for (model_name, model) in models:
            clf = model(MultinomialNB())
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_val)
            st.session_state[f"{model_name}-{encoding}-clf"] = clf
            st.write(f"{model_name}-{encoding}:")
            st.write(f" - Accuracy: {accuracy_score(y_val, y_pred)}")
            st.write(f" - Precision: {precision_score(y_val, y_pred, average='samples')}")
            st.write(f" - Recall: {recall_score(y_val, y_pred, average='samples')}")
            st.write(f" - Hamming Loss: {hamming_loss(y_val, y_pred)}")
            # TODO ROC CURVE?
    st.session_state["trained"] = True

encoding = st.selectbox(
    label='Select encoding:',
    options=encodings,
    key='encoding'
)
model_name = st.selectbox(
    label='Select model:',
    options=[model_name for (model_name, model) in models],
    key='model_name'
)

if (st.button('Deploy Model')):
    st.write("Deploying model...")
    st.session_state["deployed_encoding"] = st.session_state[f"{encoding}_vect"]
    st.session_state["deployed_clf"] = st.session_state[f"{model_name}-{encoding}-clf"]
    st.write("Model has been deployed.")

st.write('Continue to Deployed Model')
