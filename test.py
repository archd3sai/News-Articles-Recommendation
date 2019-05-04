import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#to scrape Twitter
import tweepy
from tweepy import OAuthHandler

#warning
import warnings
warnings.filterwarnings('ignore')

import configparser


#to preprocess tweets
import nltk
import string
import re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
words = set(nltk.corpus.words.words())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import os
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import nltk
import matplotlib.pyplot as plt

# Import required packages
import logging
import pyLDAvis.gensim
from numpy import array
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora.dictionary import Dictionary
from gensim.models import Phrases

import os
import sys
import pickle

config = configparser.ConfigParser()
config_path =  'model/config.ini'
config.read(config_path)

consumer_key = config['tweepy']['consumer_key']
consumer_secret = config['tweepy']['consumer_secret']
access_token = config['tweepy']['access_token']
access_secret = config['tweepy']['access_secret']


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)



def getTweets(user):
    # Get Ten Tweets from the each user
    twitterUser = api.get_user(user)

    tweets = api.user_timeline(screen_name = user, count = 10,tweet_mode='extended')
    tentweets = []
    for tweet in tweets:
        if tweet.full_text.startswith("RT @") == True:
            tentweets.append(tweet.retweeted_status.full_text)
        else:
            tentweets.append(tweet.full_text)

    return tentweets


def processTweets(tweets):

    #cleaning of tweets
    cleanedTweets = []
    for tweet in tweets:
        tw = re.sub('http\S+', '', tweet) #remove links
        tw = re.sub('RT', '', tw) #remove RT of retweet
        tw = re.sub('@[^\s]+','',tw) #remove usernames
        tw = "".join([char for char in tw if char not in string.punctuation]) #remove punctuations
        tw = tw.lower() #converting to lowercase letters
        tw = ' '.join([word for word in tw.split() if word not in (stop)]) #removing stop words
        tw = ' '.join([word for word in tw.split() if len(word)>2])
        cleanedTweets.append(tw)

    cleanedTweets = ' '.join(cleanedTweets) #joining all tweets

    #tokenization
    ProcessedTweets = nltk.word_tokenize(cleanedTweets) ####################################################

    #stemming
    ProcessedTweets = [ps.stem(word) for word in ProcessedTweets]

    #lammitization
    ProcessedTweets = [wn.lemmatize(word) for word in ProcessedTweets]

    ProcessedTweets = [word for word in ProcessedTweets if len(word)>2]

    ProcessedTweets = ' '.join(w for w in ProcessedTweets if w in words)

    return ProcessedTweets
def get_jaccard_sim(str1, str2):
    a = set(str1)
    b = set(str2)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def GetUserRec(recommend, tweets,cluster):

    articles=recommend[recommend['cluster']==cluster]
    articles.dropna(inplace=True)
    articles['corp']= processArticles(articles['article_text'])


    tweetcorp=processTweets(''.join(tweets))

    articles['jaccard']=articles['article_text'].apply(lambda x:get_jaccard_sim(tweetcorp,x))
    articles['final_score']=0.5*articles['jaccard']+0.5*articles['metric']
    ranked=articles.sort_values(by=['final_score'], ascending=False)

    return ranked
def processArticles(articles):

    #cleaning of articles
    cleanedarticles = []
    for article in articles:
        article = re.sub("[^a-zA-Z]"," ", str(article))
        article = article.lower() #converting to lowercase letters
        article = ' '.join([word for word in article.split() if word not in (stop)]) #removing stop words
        article = ' '.join([word for word in article.split() if len(word)>2])

        #tokenization
        article = nltk.word_tokenize(article)

        #stemming
        article = [ps.stem(word) for word in article]

        #lammitization
        article = [wn.lemmatize(word) for word in article]

        article = [word for word in article if len(word)>2]
        article = ' '.join(w for w in article if w in words)

        cleanedarticles.append(article)
    return cleanedarticles


usersData = pd.DataFrame({'ActiveNewsReaders':list('0')})
usersData['ActiveNewsReaders'] = sys.argv[1]
#print(usersData.head())
vfunc = np.vectorize(getTweets)
#print(sys.argv)
usersData["tweets"] = usersData['ActiveNewsReaders'].apply(lambda x: getTweets(x))

usersData["ptweets"] = usersData['tweets'].apply(lambda x : processTweets(x))

#print(usersData.shape)
tfidf_vectorizer = pickle.load(open('model/model_tfidf.pickle.dat','rb'))
km = pickle.load(open('model/model_kmeans.pickle.dat','rb'))

tfidf_matrix = tfidf_vectorizer.transform(usersData.ptweets)
cluster = km.predict(tfidf_matrix)
print(cluster[0])

recommend = pd.read_csv('model/recommend.csv')

#recommend = recommend[(recommend['cluster'] == cluster[0])]
recommend.sort_values(by='metric',ascending=False,inplace=True)
#print(usersData["tweets"].values[0])

print(GetUserRec(recommend,usersData["tweets"].values[0],cluster[0])['url'].head(10).values)
#print(recommend['url'].head(5))
