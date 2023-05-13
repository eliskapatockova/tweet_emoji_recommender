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
    return re.sub(r'\d+', '', text)

def detect_lang(text):
    try:
        return detect(text)
    except:
        return "unknown"
    
def remove_stopwords(text):
    stop = stopwords.words('english')
    no_stop = " ".join([c for c in text.split(" ") if c not in stop])
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

def word_count_encoder(df, feature):
    count_vect = CountVectorizer(max_features=10000)

    X_train_counts = count_vect.fit_transform(df[feature])
    word_count_df = pd.DataFrame(X_train_counts.toarray())
    # word_count_df = pd.DataFrame.sparse.from_spmatrix(X_train_counts)
    word_count_df = word_count_df.add_prefix('word_count_')
    # df = pd.concat([df, word_count_df], axis=1)

    word_count_df = word_count_df.reset_index(drop=True)
    df = pd.concat([df.reset_index(drop=True), word_count_df], axis=1)

    # Show confirmation statement
    st.write('{} has word count encoded {} tweets.'.format(
        feature, len(word_count_df)))

    # Store new features in st.session_state
    # st.session_state['encoded_data'] = df

    # word_encoder.append('Word Count')
    # st.session_state['word_encoder'] = word_encoder
    st.session_state['count_vect'] = count_vect

    return df

def tf_idf_encoder(df, feature):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    X = vectorizer.fit_transform(df[feature])

    tf_idf_df = pd.DataFrame(X.toarray())
    tf_idf_df = tf_idf_df.add_prefix('tf_idf_word_count_')
    tf_idf_df = tf_idf_df.reset_index(drop=True)
    df = pd.concat([df.reset_index(drop=True), tf_idf_df], axis=1)

    st.session_state['encoded_data'] = df

    # word_encoder.append('TFITDF Count')
    # st.session_state['tfidf_encoder'] = word_encoder
    st.session_state['tfidf_transformer'] = vectorizer
    
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
df = None
st.markdown("### Kaggle's tweets datasets are automatically merged and used as default.")
df = join_dataframes()
sample_df=df.sample(n = 1000)
sample_df.to_csv("./data/sample_df.csv", index=False)
st.write("If you wish to use your own dataset, please upload it below.")
st.write("Select dataset from local machine")

testing = True

if testing:
    df = pd.read_csv("./data/sample_df.csv")
else:
    if st.button('Load Dataset'):
        data = st.file_uploader(
                'Upload a Dataset', type=['csv', 'txt'])
        if (data):
            df = pd.read_csv(data)
        if df is not None:
            st.write("Loaded dataset has {} rows and {} columns.".format(
                df.shape[0], df.shape[1]))
st.session_state['data'] = df

if df is not None:
    # Display original dataframe
    st.markdown('You have uploaded the following dataset:')
    st.dataframe(df)
    st.markdown("Dataframe length: ", len(df. index))

    # Handle Text and Categorical Attributes
    st.markdown('### Preprocess Data')

    st.markdown('This button will seperate the data into text and emojis, convert text to lowercase, and remove special characters, numbers, other languages, and stopwords.')
    prep_df = None
    if (st.button('Preprocess Data')):

        prep_df = preprocess_step_1(df)
        st.write("Emojis and text have been split into separate columns, EoL, mentions, hashtags, newlines, links removed")
        
        prep_df = preprocess_step_2(prep_df)
        st.write("Special characters, numbers, and extra whitespace removed")

        prep_df = preprocess_step_3(prep_df)
        st.write("Stopwords and other languages have been removed, text has been converted to lowercase")

        prep_df = prep_df.explode("Emoji").reset_index(drop=True)
        st.write("Emoji lists turned into one emoji per tweet")

        prep_df = prep_df.dropna()

        word_encoded_df = word_count_encoder(prep_df, 'Text')
        word_encoded_df.to_csv("./data/preprocessed_sample_df.csv", index=False
                               )
        st.write("Word Encoded & Saved")

        st.dataframe(prep_df)
        st.session_state['data'] = word_encoded_df
        df = word_encoded_df

    # Inspect the dataset
    # if (prep_df is not None):
    #     st.markdown('### Inspect and visualize some interesting features')
    #     st.markdown('###### Filter tweets by emojis')
    #     data_emoji_list = list(df['Emoji'].unique())
    #     emoji = st.selectbox('Select an emoji', data_emoji_list)
    #     emoji_df = df[df['Emoji'] == emoji]
    #     st.write('There are {} tweets with the {} emoji.'.format(
    #         len(emoji_df), emoji))
    #     st.dataframe(emoji_df)

    # st.markdown('### Encode Tweet Text Data')
    # # string_columns = list(df.select_dtypes(['object']).columns)
    # # string_columns = list(df.select_dtypes(['string']).columns)
    # word_encoder = []
    # word_count_col, tf_idf_col = st.columns(2)
    # # Perform Word Count Encoding
    # with word_count_col:
    #     st.write('Word Count Encoding')
    #     word_encoded_df = None
    #     if (st.button('Word Encoder')):
    #         prep_df = st.session_state['data']
    #         st.dataframe(prep_df)
    #         st.dataframe(prep_df.reset_index(drop=True))
    #         word_encoded_df = word_count_encoder(prep_df, 'Text', "Word Encode")
    #         # Show updated dataset
    #         st.write('Word Count Encoding has been applied to {} tweets.'.format(len(word_encoded_df.index)))
    #         st.write('Updated dataset:')
    #         st.dataframe(word_encoded_df)
    #         # df = word_encoded_df
    #         st.session_state['data'] = word_encoded_df

    # # Perform TF-IDF Encoding
    # with tf_idf_col:
    #     st.write('TF-IDF Encoding')
    #     tfidf_encoded_df = None
    #     if (st.button('TF-IDF Encoder')):
    #         prep_df = st.session_state['data']
    #         tfidf_encoded_df = tf_idf_encoder(prep_df, 'Text')
    #         # Show updated dataset
    #         st.write('TF-IDF Encoding has been applied to {} tweets.'.format(len(tfidf_encoded_df.index)))
    #         st.write('Updated dataset:')
    #         st.dataframe(tfidf_encoded_df)
    #         # df = tfidf_encoded_df
    #         st.session_state['data'] = tfidf_encoded_df
        
    # st.markdown('### Encode Emojis Class Labels into Numerical Values')

    # df['Emoji_Labels'] = df['Emoji'].astype('category').cat.codes
    # st.dataframe(df)

    st.markdown('#### You have now preprocessed the dataset.')
    # st.dataframe(prep_df)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('###### Write the preprocessed data to a new csv')
        if (st.button('Write to csv')):
            df.coalesce(1).write.csv("./data/preprocessed_df", mode='overwrite', header=True)

    with col2:
        st.markdown('###### Save the preprocessed data to st.session_state')
        if (st.button('Save Dataframe')):
            st.session_state['data'] = df

    st.write('Continue to Train Model')
