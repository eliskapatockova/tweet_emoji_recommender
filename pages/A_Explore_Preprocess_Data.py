import os
import emoji
import pandas as pd
import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - <project title>")

#############################################

st.markdown('# Explore & Preprocess Dataset')

#############################################
# REF: https://stackoverflow.com/a/69423881
def space_emojis(s):
    emojis = emoji.distinct_emoji_list(s)
    return ''.join((' '+c+' ') if c in emojis else c for c in s)

df = pd.DataFrame()
for file in os.listdir("./filtered_data"):
    emoji_name = file.split(".")[0]
    df_emoji = pd.read_csv(os.path.join("./filtered_data", file))
    df_emoji = df_emoji.apply(space_emojis)
    print(df_emoji.head())
    # df_new = pd.DataFrame({"text": df_emoji["Text"], emoji_name: [1 for _ in df["Text"]]})
    # print(df_new.head())

if df is not None:
    # Display original dataframe
    st.markdown('View initial data with missing values or invalid inputs')
    st.markdown('You have uploaded the dataset.')

    st.dataframe(df)

    # Inspect the dataset
    st.markdown('### Inspect and visualize some interesting features')

    # Deal with missing values
    st.markdown('### Handle missing values')

    # Handle Text and Categorical Attributes
    st.markdown('### Handling Non-numerical Features')

    # Some feature selections/engineerings here
    st.markdown('### Remove Irrelevant/Useless Features')

    # Remove outliers
    st.markdown('### Remove outliers')

    # Normalize your data if needed
    st.markdown('### Normalize data')

    st.markdown('### You have preprocessed the dataset.')
    st.dataframe(df)

    st.write('Continue to Train Model')
