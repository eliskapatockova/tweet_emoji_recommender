import streamlit as st
import string
# from pages.A_Train_Model import preprocess_step_1, preprocess_step_2, preprocess_step_3
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - Tweet Emoji Recommendation")

#############################################

st.title('Deploy Application')
#############################################
# Helper Functions
def deploy_model(text):
    """
    Restore the trained model from st.session_state[‘deploy_model’] 
                and use it to predict the sentiment of the input data    
    Input: 
        - df: pandas dataframe with trained regression model
    Output: 
        - tweet prediction: which emoji to use
    """
    emoji_pred=None
    model=None
    if('deploy_model' in st.session_state):
        model = st.session_state['deploy_model']
        # Test model
        if(model):
            emoji_pred = model.predict(text)
    
    # return product
    return emoji_pred
#############################################

df = None
if 'data' in st.session_state:
    df = st.session_state['data']
else:
    st.write(
        '### The Tweet Emoji Recommendation Application is under construction. Coming to you soon.')

# Deploy App
if df is not None:
    st.markdown('### Introducing the Tweet Emoji Recommender <Deployment app name>')

    st.markdown('#### Some descriptions about the deployment app')

    st.markdown('### Use a trained classification method to automatically predict which emoji to use based on your tweet')

    user_input = st.text_input(
        "Enter your tweet",
        key="tweet_input",
    )

    if (user_input):
        st.write('Your input is:', user_input)

        translator = str.maketrans('', '', string.punctuation)
        # check if the feature contains string or not
        user_input_updates = user_input.translate(translator)
        
        if 'count_vect' in st.session_state:
            count_vect = st.session_state['count_vect']
            text_count = count_vect.transform([user_input_updates])
            # Initialize encoded_user_input with text_count as default
            encoded_user_input = text_count
        if 'tfidf_transformer' in st.session_state:
            tfidf_transformer = st.session_state['tfidf_transformer']
            encoded_user_input = tfidf_transformer.transform(text_count)
        
        #product_sentiment = st.session_state["deploy_model"].predict(encoded_user_input)
        tweet_prediction = deploy_model(encoded_user_input)
        st.write('The tweet is predicted to be: ', tweet_prediction)