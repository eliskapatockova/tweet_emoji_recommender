import streamlit as st
import string
import pandas as pd
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - Tweet Emoji Recommendation")

#############################################

st.title('Deploy Application')
#############################################
# Deploy App
st.markdown('### Introducing the Tweet Emoji Recommender')

st.markdown('#### Some descriptions about the deployment app')

st.markdown('### Use a trained classification method to automatically predict which emoji to use based on your tweet')

if "deployed_encoding" in st.session_state and "deployed_clf" in st.session_state:
    user_input = st.text_input(
        "Enter your tweet",
        key="tweet_input",
    )

    if (user_input):
        encoded_user_input = st.session_state["deployed_encoding"].transform([
            user_input.translate(str.maketrans('', '', string.punctuation))
        ])
            
        #product_sentiment = st.session_state["deploy_model"].predict(encoded_user_input)
        tweet_prediction = st.session_state["deployed_clf"].predict(encoded_user_input)
        st.write('The Tweet could go well with the following emojis: ')
        st.dataframe(pd.DataFrame(tweet_prediction.toarray(), columns=list('😡👀🥹😋😤😂😨😅🥵😭🫠🥳🤣😇😍🥰🥲☺️🤔😉')))
        # TODO streamline view?
else:
    st.write("Model has not been deployed yet.")