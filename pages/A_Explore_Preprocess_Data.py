import os
import emoji
import pandas as pd
import numpy as np
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
import seaborn as sns
from itertools import combinations
import plotly.express as px 
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk import ngrams
from datetime import datetime

supported_emojis = '😡👀🥹😋😤😂😨😅🥵😭🫠🥳🤣😇😍🥰🥲☺️🤔😉'

feature_lookup = {
    'emoji_😡': 'Indicator for the presence of the "angry face" emoji in the tweet.',
    'emoji_👀': 'Indicator for the presence of the "eyes" emoji in the tweet.',
    # Add more features as needed...
}

testing = True


def plot_ngrams(df, n=2):
    """
    This function creates and displays n-gram frequency plots
    """
    words = word_tokenize(" ".join(df["Text"].tolist()))
    n_grams = list(ngrams(words, n))
    n_grams_freq = Counter(n_grams)

    n_grams_df = pd.DataFrame.from_dict(n_grams_freq, orient='index').reset_index()
    n_grams_df = n_grams_df.sort_values(by=[0], ascending=False)
    n_grams_df.columns = ['N-gram', 'Frequency']
    
    plt.figure(figsize=(10,5))
    plt.bar(n_grams_df['N-gram'].apply(lambda x: ' '.join(x)).values[:20], n_grams_df['Frequency'].values[:20])
    plt.title(f'{n}-gram Frequency Plot')
    plt.xlabel('N-grams')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    st.pyplot(plt.gcf())

def create_time_series_plots(df, feature, date_col='date'): #requires time - do we have this? 
    """
    This function creates time series plots
    """
    df[date_col] = pd.to_datetime(df[date_col])
    time_series_df = df.set_index(date_col)
    time_series_df[feature].resample('D').mean().plot()
    st.pyplot(plt.gcf())

def compute_correlation(X, feature):

    correlation = X[feature].corr()
    cor_summary_statements = []

    for pair in combinations(feature, 2):  # get all combinations of pairs
        corr_value = correlation.loc[pair[0], pair[1]]
        # classify magnitude of correlation
        if abs(corr_value) > 0.7:
            magnitude = "strongly"
        elif abs(corr_value) > 0.3:
            magnitude = "moderately"
        else:
            magnitude = "weakly"
        # classify direction of correlation
        direction = "positively" if corr_value > 0 else "negatively"
        # construct summary statement
        summary = f"- Features {pair[0]} and {pair[1]} are {magnitude} {direction} correlated: {corr_value:.2f}"
        cor_summary_statements.append(summary)
    return correlation, cor_summary_statements

def display_features(df, feature_lookup):
    for feature in df.columns:
        st.write(f"{feature}: {feature_lookup.get(feature, 'No description available.')}")
    return df, df.columns.tolist()  # return the features directly from the DataFrame

def join_dataframes(Testing):
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

    if testing == True: 
        df = df.sample(n=1000)

    return df.iloc[2:, :]
 
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

# Helper Function
def user_input_features1(df, chart_type, x=None, y=None):
    """
    This function renders the feature selection sidebar  
    Input: 
        - df: pandas dataframe containing dataset
        - chart_type: the type of selected chart
        - x: features
        - y: targets
    Output: 
        - dictionary of sidebar filters on features
    """
    
    side_bar_data = []

    select_columns = []
    if (x is not None):
        select_columns.append(x)
    if (y is not None):
        select_columns.append(y)
    if (x is None and y is None):
        select_columns = list(df.select_dtypes(include='number').columns)

    for idx, feature in enumerate(select_columns):
        try:
            f = st.sidebar.slider(
                str(feature),
                float(df[str(feature)].min()),
                float(df[str(feature)].max()),
                (float(df[str(feature)].min()), float(df[str(feature)].max())),
                key=chart_type+str(idx)
            )
        except Exception as e:
            print(e)
        side_bar_data.append(f)
    return side_bar_data
    
# Helper Function
def user_input_features(df, chart_type):
    """
    This function renders the feature selection sidebar 

    Input: 
        - df: pandas dataframe containing dataset
        - chart_type: the type of selected chart
    Output: 
        - list of selected features
    """
    
    st.sidebar.markdown('### Select Features for ' + chart_type)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [st.sidebar.checkbox(col, key=chart_type+col) for col in numeric_cols]
    selected_features = [numeric_cols[i] for i in range(len(numeric_cols)) if features[i]]

    return selected_features

def create_scatterplot(df, x, y):
    """
    This function creates scatterplot for the selected features
    Input: 
        - df: pandas dataframe 
        - x: feature on x-axis
        - y: feature on y-axis
    Output: 
        - None
    """
    fig = px.scatter(df, x=x, y=y, trendline="ols")
    st.plotly_chart(fig)

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

df = join_dataframes(True) 
st.session_state['data'] = df

if df is not None:

    # Display original dataframe
    st.markdown('Raw Dataset:')
    st.dataframe(df)
    st.markdown("DataFrame Length: "+str(df.shape[0]))

    # Handle Text and Categorical Attributes
    st.markdown('### Preprocessing')
    st.markdown('This button will seperate the data into text and emojis, convert text to lowercase, and remove special characters, numbers, other languages, and stopwords.')

    if st.button('Preprocess Data'):
        prep_df = preprocess(df)
        st.write('Snippet of the result:')
        st.write(prep_df.head(10))
        st.session_state['prep_df'] = prep_df
      

st.markdown("### Data Exploration")
if st.button('Explore Data') or 'explore_data_pressed' in st.session_state:
    st.session_state.explore_data_pressed = True

    if 'prep_df' in st.session_state:
        prep_df = st.session_state['prep_df']
        prep_df = prep_df.drop(['Text'], axis=1)
        prep_df = st.session_state['prep_df']
        
        # Initialize an empty dictionary to hold the counts
        emoji_counts = {}

        # Iterate over the columns in the dataframe
        for col in prep_df.columns:
            # Check if the column is an emoji column
            if col.startswith('emoji_'):
                # Count the occurrences of the emoji and add it to the dictionary
                emoji_counts[col] = prep_df[col].sum()

        # Plot the distribution of emojis
        st.markdown('Emoji Distribution:')
        st.bar_chart(emoji_counts)

        prep_df = st.session_state['prep_df']
        st.markdown("### Word Frequency Plot")
        words = word_tokenize(" ".join(prep_df["Text"].tolist()))
        word_freq = Counter(words)
        word_freq_df = pd.DataFrame.from_dict(word_freq, orient='index').reset_index()
        word_freq_df = word_freq_df.sort_values(by=[0], ascending=False)
        word_freq_df.columns = ['Word', 'Frequency']
        st.write(word_freq_df.head(20))  # Display top 20 frequent words in dataframe

        # Plotting
        plt.figure(figsize=(10,5))
        plt.bar(word_freq_df['Word'].values[:20], word_freq_df['Frequency'].values[:20])
        plt.title('Word Frequency Plot')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=90)
        st.pyplot(plt.gcf())  # Display the plot

        st.markdown("### N-gram Frequency Plots")
        plot_ngrams(prep_df, n=2)  # bigrams
        plot_ngrams(prep_df, n=3)  # trigrams

        #st.markdown("### Time Series Plots")
        #create_time_series_plots(df, feature='your_feature', date_col='your_date_col')  # replace with your feature and date column
        
    else:
        st.write("Please preprocess the data first.")


st.markdown("### Encoding")
if st.button('word encoding'):

    prep_df = st.session_state['prep_df']
        
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