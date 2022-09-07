# -*- coding: utf-8 -*-

"""




@author: wilma


"Alumnas: Antoinette Severino 1060888 &  Wilma Ramirez 1053669"""






##pip install tweepy
import tweepy

import pandas as pd

import time





#pip install tweepy

import tweepy





from tweepy import OAuthHandler 

import os

import numpy as np

import pandas as pd

from os import path

from PIL import Image







import matplotlib.pyplot as plt 

import seaborn as sns

import re

import time

import string

import warnings



#agregamos toda los paquetes sobre textos para NLP, de analisis de lenguaje

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import *

from nltk.classify import NaiveBayesClassifier



#pip install wordcloud



from wordcloud import wordloud



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB



#pip install selenium

from selenium import webdriver

from selenium.webdriver.common.keys import Keys





#para analizar los sentimientos del texto

#pip install textblob 

from textblob import TextBlob

from textblob.sentiments import NaiveBayesAnalyzer

from textblob.np_extractors import ConllExtractor





#ignorar las advertencias



warnings.filterwarnings("ignore", category=DeprecationWarning)







from nltk import tokenize



# for showing all the plots inline

#%matplotlib inline





print("Current Working Directory " , os.getcwd())

os.chdir("C:/Users/wilma/Desktop/Maestria Estadistica Aplicada/8. Octavo Trimestre/Analisis Predictivo de Negocios")

print("Current Working Directory " , os.getcwd())



#scraping

consumer_key = "HSyx6XeIA9bY4JDtm3lUdqrpI"

consumer_secret = "nPE89PJ60s32FdqwBtOH2oF8EW7fPmGzbaAJPzVkLsvXwEvDOQ"

access_token = "1483852258504843265-KpaY60S6Fnwn8SXOaZ20Wy2Eh6xJLy"

access_token_secret = "mXeEw2ckvoGjkTJswu0URNf5s0DaFvORxQ1HoNvMdtQF4"



auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth,wait_on_rate_limit=True)



tweets = []



#scraping de una cuenta

username = 'CervPresidente'

count = 5000



 

##crear un csv con la informacion

def username_tweets_to_csv(username,count):

    try:      

        # Creation of query method using parameters

        tweets = tweepy.Cursor(api.user_timeline,id=username).items(count)

    

        # Pulling information from tweets iterable object

        tweets_list = [[tweet.created_at, tweet.id, tweet.text] for tweet in tweets]

        

        # Creation of dataframe from tweets list

        # Add or remove columns as you remove tweet information

        tweets_df = pd.DataFrame(tweets_list,columns=['Datetime', 'Tweet Id', 'Text'])



        # Converting dataframe to CSV 

        tweets_df.to_csv('{}-tweets.csv'.format(username), sep=',', index = False)



    except BaseException as e:

          print('failed on_status,',str(e))

          time.sleep(3)

          

          







##se crea data frame



import nltk

import pandas as pd

#nltk.download()

from nltk.sentiment.vader import SentimentIntensityAnalyzer



nltk.download('spanish_grammars')

nltk.download('cess_esp')



from nltk.corpus import cess_esp as cess

cess_sents = cess.tagged_sents()





analyzer = SentimentIntensityAnalyzer()



 # Creation of query method using parameters

tweets = tweepy.Cursor(api.user_timeline,id=username).items(count)

 

 # Pulling information from tweets iterable object

tweets_list = [[tweet.created_at, tweet.id, tweet.text] for tweet in tweets]

 

 # Creation of dataframe from tweets list

 # Add or remove columns as you remove tweet information

tweets_df = pd.DataFrame(tweets_list)

 





tweets_df['neg'] = tweets_df[2].apply(lambda x:analyzer.polarity_scores(x)['neg'])

tweets_df['neu'] = tweets_df[2].apply(lambda x:analyzer.polarity_scores(x)['neu'])

tweets_df['pos'] = tweets_df[2].apply(lambda x:analyzer.polarity_scores(x)['pos'])

tweets_df['compound'] = tweets_df[2].apply(lambda x:analyzer.polarity_scores(x)['compound'])









##cambiar nombre columnas

tweets_df.info()

tweets_df.columns



tweets_df.columns = ['date','id','tweet', 'neg' ,'neu','pos', 'compound']





tweets_df.to_csv("presidente_tweets_df2.csv")

#cantidad de positivos, negativos, neutros

tweets_df.sum()







neg= tweets_df['neg'].sum()

print(neg)

neu= tweets_df['neu'].sum()

print(neu)

pos= tweets_df['pos'].sum()

print(pos)





print("score neg", neg)



print("score neu", neu)



print("score pos", pos)











tweets_df['neg'].mean()

tweets_df['neu'].mean()

tweets_df['pos'].mean()









##Pre-Procesado de las palabras





#limpiando los tweets



def remove_pattern(text, pattern_regex):

    r = re.findall(pattern_regex, text)

    for i in r:

        text = re.sub(i, '', text)

    

    return text 



#limpiando los nombres

tweets_df['tidy_tweets'] = np.vectorize(remove_pattern)(tweets_df['tweet'], "@[\w]*: | *RT*")

tweets_df.head(10)





tweets_df['tweet'].head(10)





            





##limpiando los enlaces



cleaned_tweets = []



for index, row in tweets_df.iterrows():

    # Here we are filtering out all the words that contains link

    words_without_links = [word for word in row.tidy_tweets.split() if 'http' not in word]

    cleaned_tweets.append(' '.join(words_without_links))



tweets_df['tidy_tweets'] = cleaned_tweets

tweets_df.head(10)



##limpiando los tweets vacios

tweets_df = tweets_df[tweets_df['tidy_tweets']!='']

tweets_df.head()



#duplicados

tweets_df.drop_duplicates(subset=['tidy_tweets'], keep=False)

tweets_df.head()



#indez

tweets_df = tweets_df.reset_index(drop=True)

tweets_df.head()



#quitando numeros

tweets_df['absolute_tidy_tweets'] = tweets_df['tidy_tweets'].str.replace("[^a-zA-Z# ]", "")





##quitando los stopwords



# downloading stopwords corpus



nltk.download('stopwords')

nltk.download('wordnet')



stopwords_en = set(stopwords.words("english"))

stop_words_sp = set(stopwords.words('spanish'))



#en este paso concatenamos las stopwords aplicándose a una cuenta que genera contenido en inglés y español

stop_words = stop_words_sp | stopwords_en

##stop_words = stop_words_sp



from nltk.corpus import stopwords

stoplist = stop_words

print(stoplist)





cleaned_tweets = []



for index, row in tweets_df.iterrows():

    

    # filerting out all the stopwords 

    words_without_stopwords = [word for word in row.absolute_tidy_tweets.split() if not word in stop_words and '#' not in word.lower()]

    

    # finally creating tweets list of tuples containing stopwords(list) and sentimentType 

    cleaned_tweets.append(' '.join(words_without_stopwords))

    

tweets_df['absolute_tidy_tweets'] = cleaned_tweets

tweets_df.head(10)



#tokenizamos los tweets

tokenized_tweet = tweets_df['absolute_tidy_tweets'].apply(lambda x: x.split())

tokenized_tweet.head()







##de palabras a lemas

word_lemmatizer = WordNetLemmatizer()



tokenized_tweet = tokenized_tweet.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])

tokenized_tweet.head()







##tokens a oraciones

for i, tokens in enumerate(tokenized_tweet):

    tokenized_tweet[i] = ' '.join(tokens)



tweets_df['absolute_tidy_tweets'] = tokenized_tweet

tweets_df.head(10)







##preprocesamiento de términos de frases



class PhraseExtractHelper(object):

    def __init__(self):

        self.lemmatizer = nltk.WordNetLemmatizer()

        self.stemmer = nltk.stem.porter.PorterStemmer()

    

    def leaves(self, tree):

        """Finds NP (nounphrase) leaf nodes of a chunk tree."""

        for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):

            yield subtree.leaves()



    def normalise(self, word):

        """Normalises words to lowercase and stems and lemmatizes it."""

        word = word.lower()

        # word = self.stemmer.stem_word(word) # We will loose the exact meaning of the word 

        word = self.lemmatizer.lemmatize(word)

        return word



    def acceptable_word(self, word):

        """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""

        accepted = bool(3 <= len(word) <= 40

            and word.lower() not in stopwords

            and 'https' not in word.lower()

            and 'http' not in word.lower()

            and '#' not in word.lower()

            )

        return accepted



    def get_terms(self, tree):

        for leaf in self.leaves(tree):

            term = [ self.normalise(w) for w,t in leaf if self.acceptable_word(w) ]

            yield term



##reglas para identificar oraciones



sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'

grammar = r"""

    NBAR:

        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

        

    NP:

        {<NBAR>}

        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...

"""

chunker = nltk.RegexpParser(grammar)





import nltk

nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stopword = nltk.corpus.stopwords.words('english')
stopword = nltk.corpus.stopwords.words('spanish')
print(stopword[:11])





##me da error aqui! 

#creacion variable de frases

key_phrases = []

phrase_extract_helper = PhraseExtractHelper()



for index, row in tweets_df.iterrows(): 

    toks = nltk.regexp_tokenize(row.tidy_tweets, sentence_re)

    postoks = nltk.tag.pos_tag(toks)

    tree = chunker.parse(postoks)



    terms = phrase_extract_helper.get_terms(tree)

    tweet_phrases = []



    for term in terms:

        if len(term):

            tweet_phrases.append(' '.join(term))

    

    key_phrases.append(tweet_phrases)

    

key_phrases[:10]





##quitando palabras


import nltk

nltk.download('punkt')



textblob_key_phrases = []

extractor = ConllExtractor()



for index, row in tweets_df.iterrows():

    # filerting out all the hashtags

    words_without_hash = [word for word in row.tidy_tweets.split() if '#' not in word.lower()]

    

    hash_removed_sentence = ' '.join(words_without_hash)

    

    blob = TextBlob(hash_removed_sentence, np_extractor=extractor)

    textblob_key_phrases.append(list(blob.noun_phrases))



textblob_key_phrases[:10]



 

    

def generate_wordcloud(all_words):

    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=100, relative_scaling=0.5, colormap='black').generate(all_words)



    plt.figure(figsize=(14, 10))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis('off')

    plt.show()

    

    

    

##world cloud

    

pos= ' '.join([text for text in tweets_df['absolute_tidy_tweets'][tweets_df.pos >0]])

generate_wordcloud(pos)





neg= ' '.join([text for text in tweets_df['absolute_tidy_tweets'][tweets_df.neg >0 ]])

generate_wordcloud(all_words)











pos= ' '.join([text for text in tweets_df['absolute_tidy_tweets'][tweets_df.pos >0]])

generate_wordcloud(pos)





neg= ' '.join([text for text in tweets_df['absolute_tidy_tweets'][tweets_df.neg >0]])

generate_wordcloud(neg)













## hashtags comunes



def hashtag_extract(text_list):

    hashtags = []

    # Loop over the words in the tweet

    for text in text_list:

        ht = re.findall(r"#(\w+)", text)

        hashtags.append(ht)



    return hashtags



def generate_hashtag_freqdist(hashtags):

    a = nltk.FreqDist(hashtags)

    d = pd.DataFrame({'Hashtag': list(a.keys()),

                      'Count': list(a.values())})

    # selecting top 15 most frequent hashtags     

    d = d.nlargest(columns="Count", n = 25)

    plt.figure(figsize=(16,7))

    ax = sns.barplot(data=d, x= "Hashtag", y = "Count")

    plt.xticks(rotation=80)

    ax.set(ylabel = 'Count')

    plt.show()

    

    

    hashtags = hashtag_extract(tweets_df['tidy_tweets'])

    hashtags = sum(hashtags, [])





generate_hashtag_freqdist(hashtags)



