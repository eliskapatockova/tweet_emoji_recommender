import os
import emoji
import pandas as pd
import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import langdetect
from langdetect import detect
import nltk
from nltk.corpus import stopwords

def join_dataframes():
    dirname = os.path.join(os.getcwd(), 'filtered_data')
    ext = ('.csv')

    files = []
    for file in os.listdir(dirname):
        if file.endswith(ext):
            files.append(file)
        else:
            continue
    frames = []
    for f in files:
        frames.append(pd.read_csv(os.path.join(dirname, f)))
    # frames = [ process_your_file(f) for f in files ]
    # result = pd.concat(frames)
    df = pd.concat(frames)
    return df.iloc[2:, :]
 
supported_emojis = '😡👀🥹😋😤😂😨😅🥵😭🫠🥳🤣😇😍🥰🥲☺️🤔😉'
def preprocess(df_emoji):
    # get a list of emojis
    emojis = df_emoji["Text"].apply(lambda s: [emoj for emoj in emoji.distinct_emoji_list(str(s)) if emoj in supported_emojis])
    for current_emoji in supported_emojis:
        df_emoji['emoji_{emoji}'.format(emoji=current_emoji)] = emojis.apply(lambda emojis: 1 if current_emoji in emojis else 0)
    st.write(" - Emojis have been split into separate columns")
    
    nltk.download('stopwords')
    stop = stopwords.words('english')
    def cleanup_text(x):
        # remove end of line
        x = re.sub('\n', '', x)

        # remove emojis
        x = ''.join([c for c in str(x) if not emoji.is_emoji(c)])

        # remove mentions, hashtags, newlines, links
        x = re.sub('[@#][^\s]+', '', x)
        x = re.sub('&[^\s]+;', '', x)
        x = re.sub('http[^\s]+', '', x)

        # remove special characters
        x = re.sub(r"[^A-Za-z0-9@\'\`\"\_\n]", " ", x)
        # remove whitespaces
        x = re.sub(r" +", " ", x, flags=re.I)
        # remove numbers
        x = re.sub(r'\d+', '', x)

        #remove stopwords from text
        x = " ".join([c for c in x.split(" ") if c not in stop])
        # convert to lower case
        x = x.lower()
        x = x.strip()

        #remove tweets in other languages
        # df_emoji['language'] = df_emoji['Text'].apply(lambda x: detect(x))
        # df_emoji = df_emoji[df_emoji['language'] == 'en']

        return x
     
    df_emoji['Text'] = df_emoji['Text'].apply(cleanup_text)
    st.write(" - EoL, mentions, hashtags, newlines, links, special characters, numbers, and extra whitespace removed")

    df_emoji = df_emoji[df_emoji["Text"] != ""]

    return df_emoji

def word_count_encoder(df, feature):
    count_vect = CountVectorizer(max_features=2000)
    X_train_counts = count_vect.fit_transform(df[feature])
    st.session_state['word_encoded_data_vect'] = count_vect

    word_count_df = pd.DataFrame(X_train_counts.toarray())
    word_count_df = word_count_df.add_prefix('word_count_')
    word_count_df = word_count_df.reset_index(drop=True)
    df = pd.concat([df.reset_index(drop=True), word_count_df], axis=1)

    return df

def tf_idf_encoder(df, feature):
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 3))
    X = vectorizer.fit_transform(df[feature])
    st.session_state['tfidf_encoded_data_vect'] = vectorizer

    tf_idf_df = pd.DataFrame(X.toarray())
    tf_idf_df = tf_idf_df.add_prefix('word_count_')
    tf_idf_df = tf_idf_df.reset_index(drop=True)
    df = pd.concat([df.reset_index(drop=True), tf_idf_df], axis=1)
    
    return df

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - Tweet Emoji Recommendation")

#############################################

st.markdown('# Explore & Preprocess Dataset')

#############################################
# REF: https://stackoverflow.com/a/69423881


##################################### STREAMLIT APP #####################################
st.markdown("### Kaggle's tweets datasets are automatically merged and used as default.")
df = join_dataframes()

# TODO maybe make this an optional parameter in the GUI?
testing = True
if testing:
    df = df.sample(n=5000)
st.session_state['data'] = df

if df is not None:
    # Display original dataframe
    st.markdown('Raw Dataset:')
    st.dataframe(df)
    st.markdown("DataFrame Length: "+str(df.shape[0]))

    # Handle Text and Categorical Attributes
    st.markdown('### Explore & Preprocess Data')

    st.markdown('This button will seperate the data into text and emojis, convert text to lowercase, and remove special characters, numbers, other languages, and stopwords.')
    prep_df = None
    if (st.button('Preprocess Data')):
        st.write('Preprocessing Tweets...')
        prep_df = preprocess(df)
        st.write('Preprocessed Tweets. Text has been cleaned and select emojis have been one-hot encoded')
        st.dataframe(prep_df)
        st.session_state['prep_data'] = prep_df

        # TODO add histogram/other visual of emoji distribution/frequency (there's a rubric point associated here)

        st.write('Word Encoding Tweets...')
        word_encoded_df = word_count_encoder(prep_df, 'Text')
        word_encoded_df.to_csv("./data/preprocessed_df_2000_word.csv", index=False)
        st.write("Word Encoded Tweets:")
        st.dataframe(word_encoded_df)
        st.session_state['word_encoded_data'] = word_encoded_df
        
        st.write('TF-IDF Encoding Tweets...')
        tfidf_encoded_df = tf_idf_encoder(prep_df, 'Text')
        tfidf_encoded_df.to_csv("./data/preprocessed_df_2000_tfidf.csv", index=False)
        st.write('TF-IDF Encoded Tweets:')
        st.dataframe(tfidf_encoded_df)
        st.session_state['tfidf_encoded_data'] = tfidf_encoded_df

    st.write('Continue to Train Model')
