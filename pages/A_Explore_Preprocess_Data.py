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
    return df

def get_emojis(s):
    return emoji.distinct_emoji_list(str(s))

def remove_emojis(s):
    res = ''
    for c in str(s):
        if not emoji.is_emoji(c):
            res += c
    return res
    
def space_emojis(s):
    emojis = emoji.distinct_emoji_list(s)
    return ''.join((' '+c+' ') if c in emojis else c for c in s)

def remove_numbers(text):
    no_num = re.sub(r'\d+', '', text)
    return no_num

def detect_lang(text):
    try:
        return detect(text)
    except:
        return "unknown"
    
def remove_stopwords(text):
    stop = stopwords.words('english')
    no_stop = " ".join([c for c in text if c not in stop])
    return no_stop

def sepeate_emoji_lines(df):
    new_rows = []
    for _, row in df.iterrows():
        text = row['Text']
        emojis = row['Emoji']
        for emoji in emojis:
            new_rows.append({'Text': text, 'Emoji': emoji})
    return pd.DataFrame(new_rows)

def preprocess_step_1(df_emoji):
    # remove end of line
    df_emoji["Text"] = df_emoji["Text"].apply(lambda x: re.sub('\n', '', x))
    # get a list of emojis
    df_emoji["Emoji"] = df_emoji["Text"].apply(get_emojis)
    # remove mentions, hashtags, newlines, links
    df_emoji["Text"] = df_emoji["Text"].apply(remove_emojis)
    df_emoji["Text"] = df_emoji["Text"].apply(lambda x: re.sub('[@#][^\s]+', '', x))
    df_emoji["Text"] = df_emoji["Text"].apply(lambda x: re.sub('&[^\s]+;', '', x))
    df_emoji["Text"] = df_emoji["Text"].apply(lambda x: re.sub('http[^\s]+', '', x))

    return df_emoji


def preprocess_step_2(df_emoji):
    #remove special characters
    df_emoji['Text'] = df_emoji['Text'].apply(lambda x: re.sub(r"[^A-Za-z0-9@\'\`\"\_\n]", " ", x))
    #remove whitespaces
    df_emoji['Text'] = df_emoji['Text'].apply(lambda x: re.sub(r" +", " ", x, flags=re.I))
    # remove numbers
    df_emoji['Text'] = df_emoji['Text'].apply(lambda x: remove_numbers(x))
    return df_emoji


def preprocess_step_3(df_emoji):
    #remove tweets in other languages
    # df_emoji['language'] = df_emoji['Text'].apply(lambda x: detect_lang(x))
    # df_emoji = df_emoji[df_emoji['language'] == 'en']

    #remove stopwords from text
    nltk.download('stopwords')
    df_emoji['Text'] = df_emoji['Text'].apply(lambda x: remove_stopwords(x))
    # convert to lower case
    df_emoji['Text'] = df_emoji['Text'].apply(lambda x: x.lower())
    return df_emoji

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - Tweet Emoji Recommendation")

#############################################

st.markdown('# Explore & Preprocess Dataset')

#############################################
# REF: https://stackoverflow.com/a/69423881

def word_count_encoder(df, feature, word_encoder):
    count_vect = CountVectorizer()

    X_train_counts = count_vect.fit_transform(df[feature])
    word_count_df = pd.DataFrame(X_train_counts.toarray())
    word_count_df = word_count_df.add_prefix('word_count_')
    df = pd.concat([df, word_count_df], axis=1)

    # Show confirmation statement
    st.write('{} has been word count encoded from {} tweets.'.format(
        feature, len(word_count_df)))

    # Store new features in st.session_state
    st.session_state['data'] = df

    word_encoder.append('Word Count')
    st.session_state['word_encoder'] = word_encoder
    st.session_state['count_vect'] = count_vect

    return df

# Checkpoint 3
def tf_idf_encoder(df, feature, word_encoder):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    X = vectorizer.fit_transform(df['Tweet'])

    tf_idf_df = pd.DataFrame(X.toarray())
    tf_idf_df = tf_idf_df.add_prefix('tf_idf_word_count_')
    df = pd.concat([df, tf_idf_df], axis=1)

    st.write(
        '{} column has TF-IDF encoded {} tweets.'.format(feature, len(df)))
    return df

##################################### STREAMLIT APP #####################################

df = join_dataframes()

if df is not None:
    # Display original dataframe
    st.markdown('View initial data with missing values or invalid inputs')
    st.dataframe(df)

    st.markdown('You have uploaded the dataset.')
    st.dataframe(df)

    # Inspect the dataset
    st.markdown('### Inspect and visualize some interesting features')
    st.markdown('##### Filter tweets by emojis')
    #TODO

    # Handle Text and Categorical Attributes
    st.markdown('### Preprocess Data')

    st.markdown('This button will seperate the data into text and emojis, convert text to lowercase, and remove special characters, numbers, other languages, and stopwords.')
    prep_df = None
    if (st.button('Preprocess Data')):
        prep_df = preprocess_step_1(df)
        st.write("Emojis and text have been split into separate columns, EoL, mentions, hashtags, newlines, links removed")
        prep_df = preprocess_step_2(df)
        st.write("Special characters, numbers, and extra whitespace removed")
        prep_df = preprocess_step_3(df)
        st.write("Stopwords and other languages have been removed, text has been converted to lowercase")
        prep_df = pd.DataFrame(prep_df.apply(lambda x: [(x['Text'], emoji) for emoji in x['Emoji']], axis=1).sum(), columns=['Text', 'Emoji'])
        st.write("Emoji lists turned into one emoji per tweet")
        st.dataframe(prep_df)


    st.markdown('### Encode Tweet Text Data')
    # string_columns = list(df.select_dtypes(['object']).columns)
    # string_columns = list(df.select_dtypes(['string']).columns)
    word_encoder = []
    word_count_col, tf_idf_col = st.columns(2)
    ############## Task 2: Perform Word Count Encoding
    word_encoded_df = None
    if (st.button('Word Encoder')):
        word_encoded_df = word_count_encoder(df, 'Text')
        # Show updated dataset
        st.dataframe(word_encoded_df)
        df = word_encoded_df

    ############## Task 3: Perform TF-IDF Encoding
    tfidf_encoded_df = None
    if (st.button('TF-IDF Encoder')):
        tfidf_encoded_df = tf_idf_encoder(df, 'Text')
        # Show updated dataset
        st.dataframe(tfidf_encoded_df)
        df = tfidf_encoded_df
    
    st.markdown('### Encode Emojis Class Labels into Numerical Values')

    # df['Emoji_Labels'] = df['Emoji'].astype('category').cat.codes
    st.dataframe(df)

    st.markdown('### You have now preprocessed the dataset.')
    # st.dataframe(prep_df)
    st.markdown('### Write the preprocessed data to a new csv')
    if (st.button('Write to csv')):
        df.coalesce(1).write.csv("./data/preprocessed_df", mode='overwrite', header=True)

    
    if (st.button('Save Dataframe')):
        st.session_state['data'] = df

    st.write('Continue to Train Model')
