import os
import emoji
import pandas as pd
import re
import langdetect
from langdetect import detect
import nltk
from nltk.corpus import stopwords

def get_emojis(s):
    return emoji.distinct_emoji_list(str(s))

def remove_emojis(s):
    emojis = get_emojis(s)
    return ''.join('' if c in emojis else c for c in str(s))

def remove_numbers(text):
    no_num = re.sub(r'\d+', '', text)
    return no_num
def detect_lang(text):
    try:
        return detect(text)
    except:
        return "unknown"
    
def remove_stopwords(text):
    no_stop = " ".join([c for c in text if c not in stop])
    return no_stop

df = pd.DataFrame()
for file in os.listdir("./filtered_data"):
    emoji_name = file.split(".")[0]
    df_emoji = pd.read_csv(os.path.join("./filtered_data", file))
    df_emoji["Emoji"] = df_emoji["Text"].apply(get_emojis)

    # remove mentions, hashtags, newlines, links
    df_emoji["Text"] = df_emoji["Text"].apply(remove_emojis).apply(lambda x: re.sub('[@#][^\s]+', '', x)).apply(lambda x: re.sub('&[^\s]+;', '', x)).apply(lambda x: re.sub('http[^\s]+', '', x)).apply(lambda x: re.sub('\n', '', x))
    print(df_emoji.head())

    # remove numbers
    df_emoji['Text'] = df_emoji['Text'].apply(lambda x: remove_numbers(x))

    #remove tweets in other languages
    df_emoji['language'] = df_emoji['Text'].apply(lambda x: detect_lang(x))
    ataStorage = df_emoji[df_emoji['language'] == 'en']

    #remove stopwords from text
    nltk.download('stopwords')
    stop = stopwords.words('english')
    df_emoji['Text'] = df_emoji['Text'].apply(lambda x: remove_stopwords(x))

    # TODO remove special character

    # map labels to numbers
    classes = ['smiling_face_with_tear', 'thinking_face', 'smiling_face', 
    'winking_face', 'loudly_crying_face', 'smiling_face_with_hearts', 
    'smiling_face_with_heart-eyes', 'smiling_face_with_halo', 
    'grinning_face_with_sweat', 'enraged_face', 'fearful_face', 
    'hot_face', 'partying_face', 'melting_face', 'face_with_tears_of_joy', 
    'face_holding_back_tears', 'face_with_steam_from_nose', 'rolling_on_the_floor_laughing', 
    'eyes', 'face_savoring_food']

    label = range(0, 20)
    encoded_label = dict(zip(classes, label))
    df_emoji['class'] = df_emoji['class'].map(encoded_label)
