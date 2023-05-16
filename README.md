# A Recommender System of Emojis Following Tweets
### Overview
This is a machine learning application that utilizes over 400,000 data points of real Tweets to train a model that recommends the most suitable emoji to follow a newly written tweet.  


Data preprocessing includes separating emojis from text and removing special characters, tags, links, whitespace, numbers, other languages, and stopwords. Finally, the data is encoded using both word encoding and TFIDF encoding to compare performances.  


This project trains 3 different models: Binary Relevance, Classifier Chain, and Label Powerset. Based on our experimentation, Word Encoded Label Powerset performs the best.  


Finally, the model is deployed on the final page and can be tested using new potential tweets.

### Setup Instructions
1. Download the repository 
2. Install the requirements: pip install -r requirements.txt
3. Run the project: streamlit run final_project.py

